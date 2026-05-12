---
title: "Sheeprl Reference: evaluate.py"
source: tmp/sheeprl/sheeprl/algos/dreamer_v3/evaluate.py
generated: 2026-05-12
status: reference
---

# Sheeprl Reference — `evaluate.py`

> **Source**: `tmp/sheeprl/sheeprl/algos/dreamer_v3/evaluate.py` — 57 lines.
> **Purpose** (one-line): Top-level evaluation entry point — loads a checkpoint-restored state, rebuilds the world model + actor + player, then runs evaluation rollouts via `test`.
> **Imports from elsewhere in this index**: [`utils.md`](utils.md) (`test`), [`agent.md`](agent.md) (`PlayerDV3`, `build_agent`).

---

## Table of Contents

- [Lines 1–12 — Imports](#lines-112--imports)
- [Line 16 — `evaluate`](#line-16--evaluate)

---

## Lines 1–12 — Imports

```python
from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.dreamer_v3.agent import build_agent
from sheeprl.algos.dreamer_v3.utils import test
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.registry import register_evaluation
```

**What it does**: Brings in the `Fabric` accelerator wrapper, Gymnasium for action/observation space type checks, and the four sheeprl-internal helpers this entry point composes: `build_agent` to reconstruct the DreamerV3 modules from a checkpoint, `test` to drive the evaluation rollout loop, `make_env` to build a single test-mode env worker, and the logger/log-dir helpers. `register_evaluation` is the decorator that registers this function in sheeprl's algorithm registry so the CLI `sheeprl-eval algo=dreamer_v3 …` can dispatch to it. `from __future__ import annotations` defers annotation evaluation so the `Dict[str, Any]` type hints don't require runtime import resolution.

---

## Line 16 — `evaluate`

```python
@register_evaluation(algorithms="dreamer_v3")
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    env = make_env(
        cfg,
        cfg.seed,
        0,
        log_dir,
        "test",
        vector_env_idx=0,
    )()
    observation_space = env.observation_space
    action_space = env.action_space

    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

    fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
    fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)

    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    # Create the actor and critic models
    _, _, _, _, player = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["world_model"],
        state["actor"],
    )
    del _
    test(player, fabric, cfg, log_dir, greedy=False)
```

**What it does**: Registered under the `"dreamer_v3"` algorithm key, this is the function sheeprl's eval CLI calls after restoring a checkpoint into `state` (which carries `state["world_model"]` and `state["actor"]` parameter dicts). It first wires the Fabric logger (rank-0 only logs hyperparameters) and resolves the run's `log_dir`. It then builds a single test-mode env worker via `make_env(..., "test", vector_env_idx=0)()` (the trailing `()` calls the env factory the function returns) and asserts the observation space is a `gym.spaces.Dict` — DreamerV3 requires keyed observations because the encoder splits CNN vs. MLP inputs by key (`cfg.algo.cnn_keys.encoder`, `cfg.algo.mlp_keys.encoder`). It then determines `actions_dim` and the `is_continuous` flag by inspecting the action space (Box → continuous tuple, MultiDiscrete → per-head nvec list, Discrete → single-element list). Finally it calls [`build_agent`](agent.md) with the restored `world_model` and `actor` state dicts — discarding the four training-side return values (world model, actor, critic, target-critic) and keeping only `player`, the [`PlayerDV3`](agent.md) inference-time wrapper — and hands it to [`test`](utils.md) with `greedy=False` to run stochastic-policy evaluation rollouts.

---
