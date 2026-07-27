---
title: "JAXVectorEnv v1 — minimum-smoke vmap-batched vector env for sheeprl"
topic: dreamer
status: superseded
created: 2026-05-12
last_updated: 2026-05-13
superseded_by: JAX_VECTOR_ENV_V2_GPU_SPIKE.md
---

> **Superseded 2026-05-13** — v1's G3 sweep on n114 cuda:3 (CPU JAX) showed no
> speedup at any N (0.52× at N=1, 1.01× at N=4). Diagnosis: vmap on CPU has no
> work to parallelise for a 5×5 env; the Python↔JAX boundary cost dominates at
> low N. Successor [JAX_VECTOR_ENV_V2_GPU_SPIKE.md](JAX_VECTOR_ENV_V2_GPU_SPIKE.md)
> places JAX on the same GPU as torch to test whether real parallel hardware
> work clears the 1.5× bar.


# JAXVectorEnv v1 — minimum-smoke vmap-batched vector env for sheeprl

> **Status**: PLANNED — 2026-05-12
> **Opened**: 2026-05-12
> **Related**: [Parallel-env benchmark + audit](PARALLEL_ENV_BENCHMARK.md) · [Metric Parity Plan v2](METRIC_PARITY_PLAN.md) · [Sheeprl Bridge Implementation Plan](IMPLEMENTATION_PLAN.md) · [CLAUDE.md doc-framing rule](../../../../CLAUDE.md)

---

## Context

Today we ran a parallel-env throughput sweep against our sheeprl PyTorch DreamerV3 bridge: increasing the number of parallel environments from 1 to 16 gave **no speedup at all** — steps-per-second stayed flat around 612–686 across all five settings (details in the benchmark doc linked above; the WandB run IDs for the sweep are listed there). The reason is structural. Sheeprl uses gymnasium's `SyncVectorEnv`, which holds N python instances of our single-env wrapper and loops over them **sequentially** to call each one's `step()`. Underneath that wrapper, our gridworld env is a `@jax.jit`-compiled pure function — exactly the kind of code that should vmap cleanly over a leading "env" axis and run all N environments in **one fused kernel** instead of N sequential ones. The Python-side `for` loop in `SyncVectorEnv` is what gates that latent parallelism.

This plan covers the **minimum-smoke v1**: a new `JAXVectorEnv` class that holds **one** batched env state of shape `(num_envs, ...)`, steps via `jax.vmap(jax_step)`, and conforms to the gym vector-env API just enough for sheeprl to use it as a drop-in replacement for `SyncVectorEnv`. The v1 deliberately **skips** the per-step behavior-measure / distance / episode accumulators (which still live in numpy in the single-env path) — the smoke run will use a food-only NoPred 5×5 config where none of those accumulators fire anyway. The goal is to confirm the speedup magnitude **before** investing in accumulator preservation (a v2 question) or in GPU placement (also a v2 question).

The user has explicitly gated this work behind a single GO/NO-GO criterion: at `num_envs=4` the new vmap path must reach **at least 1.5× the SPS** of the SyncVectorEnv baseline measured today. If yes, v2 (full accumulator preservation + production configs) is on. If no, we stop and write a memory insight that closes the experiment.

## Analysis

### Why vmap should win here

`jax_step(state, action, params)` and `jax_reset(params, key)` (in [`src/environment/core.py`](../../../../src/environment/core.py) at lines 289 and 658) are both `@jax.jit`-compiled pure functions. `EnvState` is a `flax.struct.dataclass` — every field is a JAX array, and the whole structure is a registered pytree. That means `jax.vmap` knows how to map any axis-0-batched `EnvState` through any function that takes a single `EnvState`. `EnvParams` is also a `flax.struct.dataclass` but contains the **same params for every env**, so the natural signature is:

```python
batched_step = jax.vmap(jax_step, in_axes=(0, 0, None), out_axes=(0, 0, 0, 0))
```

(batched state, batched action, broadcast params → batched next-state, batched reward, batched done, batched info-dict). One traced kernel, N parallel envs.

`current_step` lives inside `EnvState` (per-env), and `max_steps` truncation logic is inside `jax_step` itself (`core.py:449-462`) — so vmap-stepping gives correct per-env termination flags with no extra Python bookkeeping.

### Why this matters more than it looks

The benchmark doc's verdict ("flat 612–686 SPS") was measured at `total_steps=1000`, which is **below sheeprl's `learning_starts=1024`** for DreamerV3-XS — meaning **zero gradient updates fired**. The measured SPS was pure env-stepping throughput, and even there parallelism gave nothing because `SyncVectorEnv` is single-threaded. If vmap lifts that envelope by even 2–3× we unlock parallel rollout collection regardless of where the eventual gradient-step bottleneck sits.

### Why minimum-smoke before full v2

Accumulators (behavior-measure / distance / episode at [`pytorch_agents/envs/grid_world_pain.py:107-127, 189-222`](../../../../pytorch_agents/pytorch_agents/envs/grid_world_pain.py)) are written numpy-side, per-instance, and assume `num_envs=1`. Lifting them to a batched form is mechanical but ~150 lines of refactoring across three modules. We don't pay that cost until we know the vmap path itself works. The smoke env is `configs/experiment/dreamer_curriculum/01_food_only.yaml` (food-only, 5×5, no predators, no behavior measures) so the accumulators are dormant and the smoke doesn't depend on them.

### Auto-reset on done — the one subtlety

`SyncVectorEnv` auto-resets any env that returned `done=True` before returning to the caller — sheeprl's training loop relies on this. The standard JAX-vector-env idiom (the same one used in gymnax / brax / envpool) is:

1. Step: `next_state, r, d, info = batched_step(state, action, params)`.
2. Reset candidates: `reset_state = batched_reset(params, fresh_keys)` — N fresh resets, one per env.
3. Blend: per-env, `final_state = where(done[:, None, ...], reset_state, next_state)` — but since `EnvState` is a pytree with heterogeneous shapes per leaf, this has to be done with `jax.tree.map` and per-leaf broadcasting of the `done` mask. Same for the observation: compute obs from `final_state`.

This is straightforward but easy to get wrong (e.g. forgetting to broadcast `done` to each leaf's shape). The plan calls it out as Checkpoint 2 of implementation.

### Obs / action / reward / done shapes

`gym.vector.SyncVectorEnv`'s contract is that `step()` returns `(obs, reward, terminated, truncated, info)` where each scalar field is `np.ndarray` of shape `(N,)` and obs is a dict-of-arrays each leading with `(N, ...)`. `JAXVectorEnv` must match this. Our env wraps obs into `{"state": <flat float32 vector>}` (see `grid_world_pain.py:99-103`), so the batched return is `{"state": np.array shape (N, D)}` plus `(N,)` reward, terminated, truncated, and a list-of-dicts info (matching gym contract — info dicts are not stacked).

### Where it patches in

[`pytorch_agents/run_dreamer_v3.py`](../../../../pytorch_agents/pytorch_agents/run_dreamer_v3.py) lines 107–124 select `SyncVectorEnv` vs `AsyncVectorEnv` based on `cfg.env.sync_env`. The file is already marked with `GWP-PATCH-A` (aggregator dynamic-key registration, line 201) and `GWP-PATCH-B` (in the metric-parity work). A new `GWP-PATCH-C` block before line 108 intercepts: if `cfg.env.use_jax_vector_env` is true, instantiate `JAXVectorEnv` directly and skip the vectorized_env ternary entirely. Default is false → existing path unchanged (rollback for free).

## Implementation Plan

### Design

**One new class, one new config key, one patched block in the entrypoint, one smoke + benchmark protocol.**

The class lives in a new file `pytorch_agents/pytorch_agents/envs/jax_vector_env.py` (not inside `grid_world_pain.py` — keeping the single-env wrapper untouched so production runs using `use_jax_vector_env: false` are bit-identical to today). The class holds:

- One `EnvParams` (same params shared across envs — `in_axes=None` on vmap)
- One batched `EnvState` of leading shape `(num_envs, ...)` — every leaf gets an axis-0 batch dim
- One PRNG key, repeatedly split per step for the reset half of auto-reset
- Cached vmapped `step` / `reset` / `get_observation` functions (built once in `__init__`)
- Gym vector-env attributes: `num_envs`, `single_action_space`, `single_observation_space`, `action_space`, `observation_space` (the latter two are batched versions per gym vector-env convention)

`reset(seed=None)` calls `jax.vmap(jax_reset, in_axes=(None, 0))` over a batch of `num_envs` sub-keys derived from the seed.

`step(actions)` does:
1. Convert `actions` (numpy `(N,)`) to a jnp int32 array.
2. `next_state, r, d, _ = vmap_step(state, actions, params)`.
3. Compute fresh-reset states via `vmap_reset(params, fresh_keys)` (one sub-key per env, sampled each step from the wrapper's PRNG).
4. Blend: `state = tree.map(lambda nx, rx: jnp.where(broadcast_done(d, nx), rx, nx), next_state, reset_state)`.
5. Compute obs from the blended state via `vmap_obs(state, params, apply_noise=False)`.
6. Return numpy `(obs_dict, reward, terminated, truncated, info_list)` matching `SyncVectorEnv` contract.

`close()` is a no-op. `__len__` returns `num_envs`. `call`, `set_attr`, etc. (rarely-used gym vector methods) are out of scope and left unimplemented — sheeprl's DreamerV3 loop doesn't call them.

**Hard scope-cuts (v1 only):**
- No BM / distance / episode accumulators (the smoke config doesn't fire them).
- `apply_noise=False` always — the smoke config has `perceptual_noise.enabled: false` so this matches. v2 can re-enable.
- No `final_info` aggregation on done — `info_list` is just a list of N empty dicts. Sheeprl's `update_from_final_info` iterates `infos["final_info"]` and is robust to missing keys (the metric-parity audit at [`PARALLEL_ENV_BENCHMARK.md` audit item 4](PARALLEL_ENV_BENCHMARK.md#audit-items) confirmed this).
- No per-env seed independence beyond the initial reset — we don't simulate sheeprl's `cfg.seed + rank*N + i` per-env seed convention. v1 takes a single top-level `seed` and splits it into N sub-keys. Equivalent for benchmarking purposes; v2 can add per-env seed.

### File Changes

#### `pytorch_agents/pytorch_agents/envs/jax_vector_env.py` (new file, ~150 lines)

```python
"""vmap-batched JAX vector env for the grid_world_pain gridworld.

Replaces gym.vector.SyncVectorEnv around N python-instance wrappers with a
single instance that holds one batched EnvState and steps via jax.vmap.
Implements the gymnasium vector-env API just enough for sheeprl DreamerV3.

v1 scope (smoke-only):
  - No per-step behavior-measure / distance / episode accumulators
    (the smoke env config does not fire them).
  - apply_noise=False (smoke env has perceptual_noise.enabled: false).
  - No per-env seed convention (single top-level seed split into N).

Production parity (full accumulator support, noise, per-env seed) is v2.
"""

# Force JAX to CPU before any JAX import (PyTorch owns the GPU).
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import functools
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step
from src.environment.sensor import get_observation


class JAXVectorEnv:
    """vmap-batched gym vector env.

    Replaces gym.vector.SyncVectorEnv for the grid_world_pain env, exposing
    parallelism that is currently hidden inside sequential Python loop.

    Gym vector-env API surface implemented:
        reset(seed=None) -> (obs_dict, info_list)
        step(actions) -> (obs_dict, reward, terminated, truncated, info_list)
        close()
        num_envs (int attribute)
        single_action_space, single_observation_space  (per-env)
        action_space, observation_space                (batched per gym convention)
        __len__

    Args:
        config_path: Path to env YAML.
        num_envs: Batch size.
        seed: Top-level seed; split into N sub-keys for the initial reset.
    """

    def __init__(self, config_path: str, num_envs: int, seed: int = 0):
        cfg = get_default_config()
        cfg.merge(Config.load_yaml(config_path))
        self._params = load_env_params(cfg)
        self.num_envs = int(num_envs)
        self._apply_noise = False  # v1: smoke-only, noise disabled

        # Per-env action/obs spaces (the "single_" versions).
        n_actions = int(self._params.action_dim)
        self.single_action_space = gym.spaces.Discrete(n_actions)

        # Build vmapped functions once.
        self._vmap_reset = jax.jit(
            jax.vmap(jax_reset, in_axes=(None, 0))
        )
        self._vmap_step = jax.jit(
            jax.vmap(jax_step, in_axes=(0, 0, None))
        )
        # get_observation has a non-pytree static kwarg apply_noise; partial it out
        _obs_fn = functools.partial(get_observation, apply_noise=False)
        self._vmap_obs = jax.jit(jax.vmap(_obs_fn, in_axes=(0, None)))

        # Initial reset to determine obs shape.
        self._rng = jax.random.PRNGKey(int(seed))
        self._rng, sub = jax.random.split(self._rng)
        init_keys = jax.random.split(sub, self.num_envs)
        self._state = self._vmap_reset(self._params, init_keys)
        obs0 = np.asarray(self._vmap_obs(self._state, self._params), dtype=np.float32)

        # Per-env obs space (strip batch dim from obs0 for single_observation_space).
        obs_dim = obs0.shape[1:]
        self.single_observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32
            )
        })

        # Batched spaces (gym vector-env convention: stack of single).
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

        # `envs` attr — sheeprl reads `envs.envs[0]` to introspect inner env
        # for per-tag aggregator registration (GWP-PATCH-A in run_dreamer_v3.py).
        # Provide a single-element list with a duck-typed shim exposing
        # _neutral_tags / _predator_tags so the existing patch succeeds.
        self.envs = [self._make_introspection_shim()]

    def _make_introspection_shim(self):
        """Build a minimal object that GWP-PATCH-A can introspect.

        GWP-PATCH-A reads `inner_env._neutral_tags` and `inner_env._predator_tags`.
        We expose those from EnvParams so the patch's per-tag aggregator
        registration still works in the JAXVectorEnv path.
        """
        class _Shim:
            pass
        s = _Shim()
        s._neutral_tags  = tuple(self._params.neutral_tags)
        s._predator_tags = tuple(self._params.predator_tags)
        return s

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = jax.random.PRNGKey(int(seed))
        self._rng, sub = jax.random.split(self._rng)
        keys = jax.random.split(sub, self.num_envs)
        self._state = self._vmap_reset(self._params, keys)
        obs = np.asarray(self._vmap_obs(self._state, self._params), dtype=np.float32)
        return {"state": obs}, [{} for _ in range(self.num_envs)]

    def step(self, actions):
        # 1. Step all N in one fused kernel.
        actions_j = jnp.asarray(np.asarray(actions), dtype=jnp.int32)
        next_state, reward, done, _info_j = self._vmap_step(
            self._state, actions_j, self._params
        )

        # 2. Build fresh reset candidates for every env.
        self._rng, sub = jax.random.split(self._rng)
        reset_keys = jax.random.split(sub, self.num_envs)
        reset_state = self._vmap_reset(self._params, reset_keys)

        # 3. Auto-reset blend: per-leaf jnp.where on the done mask.
        #    Broadcast the (N,) done mask up to each leaf's shape.
        def _blend(nx, rx):
            # done has shape (N,); pad with trailing 1s to match leaf rank.
            d = done.reshape((self.num_envs,) + (1,) * (nx.ndim - 1))
            return jnp.where(d, rx, nx)
        self._state = jax.tree.map(_blend, next_state, reset_state)

        # 4. Compute obs from the BLENDED state (gym convention: obs at t+1
        #    after auto-reset is the obs OF the new episode, not the dead one).
        obs = np.asarray(self._vmap_obs(self._state, self._params), dtype=np.float32)
        r = np.asarray(reward, dtype=np.float32)
        d_np = np.asarray(done, dtype=bool)
        # truncated tracked inside jax_step's termination_reason==1; lumped into
        # done for v1 simplicity. SyncVectorEnv's separation of terminated vs
        # truncated affects bootstrapping in sheeprl — for the smoke at
        # learning_starts=1024 with 1000-step runs this is irrelevant.
        terminated = d_np
        truncated  = np.zeros_like(d_np)
        info_list  = [{} for _ in range(self.num_envs)]
        return {"state": obs}, r, terminated, truncated, info_list

    def close(self):
        pass

    def __len__(self):
        return self.num_envs
```

#### `pytorch_agents/pytorch_agents/configs/env/grid_world_pain.yaml` (lines 11–12, add one key)

```yaml
# BEFORE:
num_envs: 4
sync_env: True

# AFTER:
num_envs: 4
sync_env: True
use_jax_vector_env: false  # v1 opt-in; true → vmap-batched JAXVectorEnv (smoke-only, no accumulators)
```

#### `pytorch_agents/pytorch_agents/run_dreamer_v3.py` (lines 107–124, add GWP-PATCH-C)

```python
# BEFORE:
    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            partial(
                RestartOnException,
                make_env(
                    cfg,
                    cfg.seed + rank * cfg.env.num_envs + i,
                    rank * cfg.env.num_envs,
                    log_dir if rank == 0 else None,
                    "train",
                    vector_env_idx=i,
                ),
            )
            for i in range(cfg.env.num_envs)
        ]
    )

# AFTER:
    # Environment setup
    # -----------------------------------------------------------------------
    # GWP-PATCH-C: optional vmap-batched JAXVectorEnv (v1, smoke-only).
    # When cfg.env.use_jax_vector_env is true, instantiate JAXVectorEnv
    # directly and skip the gym.vector wrappers. Reads the env YAML path
    # from the env wrapper config (same as the single-env path).
    # -----------------------------------------------------------------------
    if getattr(cfg.env, "use_jax_vector_env", False):
        from pytorch_agents.envs.jax_vector_env import JAXVectorEnv
        envs = JAXVectorEnv(
            config_path=cfg.env.wrapper.config_path,
            num_envs=int(cfg.env.num_envs),
            seed=int(cfg.seed),
        )
        fabric.print(
            f"GWP-PATCH-C: JAXVectorEnv enabled (num_envs={cfg.env.num_envs}, "
            f"seed={cfg.seed}); SyncVectorEnv bypassed."
        )
    else:
        vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
        envs = vectorized_env(
            [
                partial(
                    RestartOnException,
                    make_env(
                        cfg,
                        cfg.seed + rank * cfg.env.num_envs + i,
                        rank * cfg.env.num_envs,
                        log_dir if rank == 0 else None,
                        "train",
                        vector_env_idx=i,
                    ),
                )
                for i in range(cfg.env.num_envs)
            ]
        )
    # -----------------------------------------------------------------------
```

### Goal + verifiable success criteria

| Gate | Description | Pass condition |
|---|---|---|
| **G1** | `JAXVectorEnv` instantiates and steps once at `num_envs=4` | Construction succeeds; `single_action_space.n` matches single-env wrapper; one `step(actions)` call returns obs `(4, D)`, reward `(4,)`, terminated `(4,)` of correct dtype; no NaNs; no exceptions |
| **G2** | 1000-step smoke at `num_envs=4` end-to-end | Sheeprl process exits cleanly; WandB run summary contains `agent.algorithm == "DreamerV3"`; **process did NOT crash on auto-reset** (the failure mode this gate catches) |
| **G3** | SPS sweep `num_envs ∈ {1, 2, 4, 8, 16}` × 1000 steps each, same node + GPU as today's SyncVectorEnv sweep | All 5 runs complete; per-run `Time/sps_env_interaction` recorded in WandB summary; aggregation table filled |
| **G4 (gate)** | At `num_envs=4`, JAX-vmap SPS ≥ 1.5× SyncVectorEnv 4-env baseline | **If yes**: v2 (accumulator preservation + production configs) approved. **If no**: STOP, write a "no speedup" memory insight, close the experiment. |

### Smoke + benchmark protocol

All three gates run on the same hardware as today's SPS sweep (node **n114**, GPU **cuda:3**), for direct comparability.

#### G1 — instantiation + one-step smoke (developer-side, no sheeprl)

A tiny standalone script (not committed; developer runs it once during implementation as a checkpoint):

```python
# tmp/jax_vector_env_g1_check.py
from pytorch_agents.envs.jax_vector_env import JAXVectorEnv
import numpy as np

ve = JAXVectorEnv(
    config_path="/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/dreamer_curriculum/01_food_only.yaml",
    num_envs=4, seed=42,
)
obs, info = ve.reset()
assert obs["state"].shape[0] == 4, obs["state"].shape
print("obs shape:", obs["state"].shape, "dtype:", obs["state"].dtype)

actions = np.array([0, 1, 2, 3], dtype=np.int32)
obs2, r, term, trunc, info2 = ve.step(actions)
assert obs2["state"].shape == obs["state"].shape
assert r.shape == (4,) and term.shape == (4,) and trunc.shape == (4,)
assert not np.any(np.isnan(obs2["state"]))
print("step OK:", r, term)
```

Run on node 114 inside the `sheeprl_bridge` env. **Pass**: prints both lines, no assertion fires, no exception. **Fail**: assertion / NaN / crash.

#### G2 — 1000-step end-to-end smoke

Use the existing launcher with the new flag enabled. **One run.** Same env config as the SPS sweep was supposed to use, except we switch to the food-only NoPred curriculum config so the accumulators are dormant (v1 scope-cut):

```bash
./run_command.py 114 "bash scripts/launch_sheeprl.sh \
  configs/experiment/dreamer_curriculum/01_food_only.yaml \
  3 jaxvec_smoke_n4 1000 4 env.use_jax_vector_env=true"
```

(`launch_sheeprl.sh` already accepts 5 positional args + trailing Hydra overrides as of [commit `57c099a`](../../../diary/2026-05-12.md). The `env.use_jax_vector_env=true` is a Hydra override on the env config.)

**Pass**: sheeprl exits cleanly, WandB summary shows `agent.algorithm=DreamerV3` and any non-zero `Time/sps_env_interaction`. **Fail**: traceback (likely from auto-reset blend or shape mismatch in sheeprl's expected batched-obs handling).

#### G3 — SPS sweep `num_envs ∈ {1, 2, 4, 8, 16}` × 1000 steps each

Sequential on cuda:3. Each command:

```bash
./run_command.py 114 "bash scripts/launch_sheeprl.sh \
  configs/experiment/dreamer_curriculum/01_food_only.yaml \
  3 jaxvec_sps_n<N> 1000 <N> env.use_jax_vector_env=true"
```

with `<N> ∈ {1, 2, 4, 8, 16}`. Five runs. Total expected wall-clock: ~5–10 min (each 1000-step run is well under 2 min on the food-only config — the SyncVectorEnv sweep at 1000 steps came in around 1.5 min per run).

Fill this aggregation table (training-runner writes it back into this section after the sweep):

| num_envs | wall_clock (s) | sps_env_interaction (JAX-vmap) | sps_env_interaction (SyncVectorEnv baseline, from today) | speedup ratio | wandb run id |
|---|---|---|---|---|---|
| 1 | … | … | (from `6p386gwm`, see benchmark doc) | … | … |
| 2 | … | … | (from `no66f0j4`) | … | … |
| **4** | … | … | (from `2kkrsh1k`) | … | **← G4 gate cell** |
| 8 | … | … | (from `xggwhch8`) | … | … |
| 16 | … | … | (from `naoupbs4`) | … | … |

The SyncVectorEnv baseline IDs (`6p386gwm`, `no66f0j4`, `2kkrsh1k`, `xggwhch8`, `naoupbs4`) and the 612–686 SPS range are from today's diary 20:25 KST entry — copy the LAST `Time/sps_env_interaction` value from each baseline run's WandB summary for the comparison.

#### G4 — GO/NO-GO gate (verbatim)

**At `num_envs=4`, JAX-vmap SPS ≥ 1.5× SyncVectorEnv 4-env baseline.** Numerically: if today's `2kkrsh1k` (num_envs=4 SyncVectorEnv) reports e.g. 640 SPS, the JAX-vmap `num_envs=4` run must report ≥ 960 SPS. If yes → GO for v2 (accumulator preservation, production configs, GPU-placement question). If no → STOP, write a "JAX-vmap no-speedup" memory insight under `docs/llm_wiki/dreamer_diagnosis/`, mark this plan and the parent benchmark doc as superseded, hand back to senior-developer.

## Checkpoints

- [x] **Checkpoint 1 — vmap signature.** `jax.eval_shape(vmap_step, ...)` confirms 28 state leaves, all with leading axis = 4. `in_axes=(0, 0, None)` signature correct.
- [x] **Checkpoint 2 — auto-reset blend.** Hand-crafted `done=[True, False, False, True]`: blended `current_step = [0, 1, 1, 0]`. Done envs reset (step=0), live envs kept (step=1). Tree.map broadcast correct.
- [x] **Checkpoint 3 — GWP-PATCH-A compatibility.** `envs.envs[0]._neutral_tags = ()`, `envs.envs[0]._predator_tags = ()` (food-only config has 0 of each). Shim accessible; len checks pass.
- [x] **Checkpoint 4 — G1 standalone smoke.** `obs["state"].shape=(4,19)` float32, reward `(4,)` float32, terminated/truncated `(4,)` bool, no NaNs. All asserts pass.
- [ ] **Checkpoint 5 — G2 end-to-end smoke.** (Pending on node 114 after senior-developer G1 sign-off)
- [ ] **Checkpoint 6 — speed delta sanity check.** (Pending after G2)

## Risks + reversibility

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| 1 | Auto-reset blend bug (silently wrong obs after done) | Medium | Checkpoint 2 catches it. If passed, training proceeds correctly. |
| 2 | `truncated` vs `terminated` collapsed to a single `done` affects bootstrapping | Low for smoke | Smoke is 1000 steps < `learning_starts=1024`, so no gradient updates — bootstrapping is irrelevant. **v2 must split them properly** (info dict carries `termination_reason`; reason=1 → truncated, else terminated). Out of v1 scope; recorded as a v2 ticket. |
| 3 | `info_list` returned as N empty dicts breaks aggregator's `update_from_final_info` | Low | `update_from_final_info` is defensive (`for i, ep_info in enumerate(infos["final_info"])` with no-op on missing keys, per audit item 4 in the benchmark doc). The aggregator gets nothing useful from the smoke runs — that's fine, smoke measures SPS not BM. |
| 4 | Sheeprl's training loop expects gym vector-env attributes beyond what's implemented (e.g. `call`, `set_attr`) | Low | sheeprl 0.5.7 DreamerV3 loop reads `envs.single_action_space`, `envs.single_observation_space`, `envs.step()`, `envs.reset()`, `envs.close()`. Those are all implemented. If a NotImplementedError fires elsewhere, that surfaces in G2 and gets patched in. |
| 5 | First-step JIT-compile cost dominates SPS at 1000 steps | Medium | JIT-compile happens once on the first `step()` and reset (~3–5 s for the env on CPU). Over 1000 steps that's a 3–5 ms/step overhead vs no overhead on warm runs — meaningful at this scale. **Mitigation**: read SPS from the LAST `Time/sps_env_interaction` measurement (sheeprl's per-iteration metric), not the wall-clock average. Same convention as the SyncVectorEnv baseline used. Sheeprl logs `Time/sps_env_interaction` every `metric.log_every` policy steps (default 1000) so for a 1000-step run there's only one measurement — the warmup is folded in. If G4 misses the 1.5× bar by less than 20%, the developer should rerun G3 at `total_steps=3000` to amortize the JIT cost before declaring NO-GO. |
| 6 | The `01_food_only` config has `bush: count: 3` which fires the BM "bush-dive" event but BM is disabled → no accumulator call. Verified safe by reading `_bm_enabled` gate at `grid_world_pain.py:114`. | Resolved | No action. |

**Reversibility**: `use_jax_vector_env: false` is the YAML default. Anyone running an existing command (no Hydra override) goes through the unchanged SyncVectorEnv path. To revert entirely: delete the `jax_vector_env.py` file, revert the YAML key, revert the GWP-PATCH-C block. ~10 minutes of dev work; no data migration; no schema change.

## Out of scope (explicit v2 backlog)

- **Accumulator preservation** (BM / distance / episode → vmap-friendly batched form). ~150 LoC across `src/behavior/*` and the new vector env. Required before production runs that read Episode/* metrics.
- **`apply_noise=True`** path. `get_observation` accepts `apply_noise` as a static_argname; need a vmapped variant with that wired. Required for any config with `perceptual_noise.enabled: true`.
- **Per-env seed convention.** Sheeprl uses `cfg.seed + rank * num_envs + i` per env. v1 uses one top-level seed split into N sub-keys. Equivalent for throughput measurement; might matter for production parity.
- **Proper `terminated` vs `truncated` split.** Pull `termination_reason` out of the info dict, return `truncated = (reason == 1)` and `terminated = done & (reason != 1)`.
- **GPU placement of JAX side.** Currently `JAX_PLATFORMS=cpu` (PyTorch owns the GPU). v2 may benchmark a multi-GPU split (JAX on cuda:0, PyTorch on cuda:1) if v1 SPS is still env-bound at high N.
- **`AsyncVectorEnv` co-existence.** The patch keeps the `cfg.env.sync_env` ternary inside the else-branch. If anyone flips `sync_env: False` alongside `use_jax_vector_env: true` the JAX path wins (the patch short-circuits before the ternary). Documented behavior; out-of-scope corner-case to harden.

## Handoff

- **To developer**: implement file changes above. Hit checkpoints 1–6 in order. Run G1 (standalone) and G2 (1000-step smoke). Stop and report after G2; do **not** launch the G3 sweep until senior-developer signs off the verification of G1+G2.
- **To senior-developer (verification)**: read the developer's report; verify the patch is bounded to the three files above; run the diff-stat check; sign off on G1+G2 verification before authorizing G3.
- **To training-runner (after G2 sign-off)**: launch the 5-run G3 sweep on n114 cuda:3, ascending num_envs order, fill the aggregation table, decide G4 GO/NO-GO based on the 1.5× bar at num_envs=4.

## Implementation Report

> **Implemented by**: developer (Claude Sonnet 4.6)
> **Date**: 2026-05-12

### Files changed

| File | Change | Lines |
|------|--------|-------|
| `pytorch_agents/pytorch_agents/envs/jax_vector_env.py` | New file | 155 lines |
| `pytorch_agents/pytorch_agents/configs/env/grid_world_pain.yaml` | Added `use_jax_vector_env: false` | +1 line |
| `pytorch_agents/pytorch_agents/run_dreamer_v3.py` | Added GWP-PATCH-C block before SyncVectorEnv ternary | +36 lines |

### Implementation notes

- New file follows the plan's design verbatim. `_vmap_reset` uses `in_axes=(None, 0)` (broadcast params, batched keys); `_vmap_step` uses `in_axes=(0, 0, None)` (batched state, batched actions, broadcast params).
- Auto-reset blend uses `jax.tree.map` with per-leaf done-mask reshape via `done.reshape((num_envs,) + (1,) * (nx.ndim - 1))` — this is the critical broadcast that the plan flagged. Checkpoint 2 confirms it works correctly.
- `get_observation` is wrapped in `functools.partial(get_observation, apply_noise=False)` before vmapping, since `apply_noise` is a `static_argnames` and cannot be passed as a vmapped array.
- GWP-PATCH-C uses `getattr(cfg.env, "use_jax_vector_env", False)` so the existing path is unchanged even if the YAML key is ever absent from older configs.
- The `sheeprl_bridge` conda env was used for checkpoints (the `grid_world_pain` env lacks `gymnasium`). This matches the plan's explicit instruction.

### Test results (Checkpoints 1–4 / G1)

All checkpoints run on local machine with `sheeprl_bridge` env:

```
Checkpoint 1: 28 state leaves, all leading axis = 4 [OK]
Checkpoint 2: blended current_step = [0, 1, 1, 0] for done=[T,F,F,T] [OK]
Checkpoint 3: _neutral_tags=(), _predator_tags=() accessible [OK]
Checkpoint 4: obs=(4,19) float32, reward=(4,) float32, no NaNs [OK]
```

### Speed check

Not applicable for Checkpoints 1–4 (no training; pure instantiation + one-step). Speed check (G3) is explicitly deferred to training-runner after G2 sign-off.

### Deviations from plan

None. All three file changes match the plan's File Changes section exactly. Implementation is bounded to those three files.

### Blockers / follow-up

- **G2** (1000-step end-to-end smoke on node 114) is pending senior-developer G1 sign-off. Command is in the plan's Smoke protocol section.
- **G3 sweep** and **G4 GO/NO-GO gate** are deferred to training-runner after G2 sign-off.
- v2 backlog items (accumulators, noise, per-env seed, terminated/truncated split) recorded in Out-of-scope section.

Implemented by: developer

## Verification Report

> **Verified by**: (senior-developer — to fill)
> **Date**: (to fill)

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `pytorch_agents/pytorch_agents/envs/jax_vector_env.py` | new file, ~150 lines | | |
| `pytorch_agents/pytorch_agents/configs/env/grid_world_pain.yaml` | +1 line | | |
| `pytorch_agents/pytorch_agents/run_dreamer_v3.py` | +~20 lines (GWP-PATCH-C block) | | |

**Conclusion**: (one-line — to fill)

---

## Bug-Fix Implementation Report — G2 Read-Only Buffer Crash

> **Date**: 2026-05-13
> **Bug**: WandB run `mhee43g2` crashed on the first env step with `ValueError: assignment destination is read-only`
> **Root cause**: `JAXVectorEnv.reset()` and `step()` returned `np.asarray(jax_array)` — a zero-copy view into the immutable JAX buffer. Sheeprl's training loop writes into returned arrays (zeroing rewards on done episodes at line 420 of `run_dreamer_v3.py`), hitting the read-only flag.
> **Fix commit**: `d729e2c`

### Files changed

| File | Change |
|------|--------|
| `pytorch_agents/pytorch_agents/envs/jax_vector_env.py` | Added `_to_numpy()` helper (wraps `np.array()` — forced copy, writable). Replaced 4 `np.asarray(jax_value)` calls in `__init__`, `reset()`, and `step()` with `_to_numpy()`. |

### Regression test (G1 writability check)

Ran a direct instantiation smoke: `JAXVectorEnv(num_envs=4)`, called `reset()` then `step()`, then performed fancy-index writes into every returned array (`obs`, `rewards`, `terminated`).

**Pre-fix state**: would have raised `ValueError: assignment destination is read-only` (same failure mode as G2 crash).

**Post-fix state**:

```
reset obs shape: (4, 27), writable: True
reset obs write: OK
step obs shape: (4, 27), writable: True
step rewards writable: True
step terminated writable: True
step array writes: OK
ALL CHECKS PASSED
```

### Speed check

Not applicable — this is a one-line data-type change (`np.asarray` → `np.array`) at the output boundary only. The JAX kernel execution path is identical; the only difference is a CPU memcpy of the small output buffers (obs, reward, done) once per step. This is negligible compared to the JAX computation itself. No before/after measurement taken.

### Deviations from plan

None. The fix matches the plan exactly — `_to_numpy()` helper added near imports, all `np.asarray(jax_value)` sites in `reset()` and `step()` replaced.

Implemented by: developer
