---
title: "Sheeprl Reference: utils.py (dreamer_v3)"
source: tmp/sheeprl/sheeprl/algos/dreamer_v3/utils.py
generated: 2026-05-12
status: reference
---

# Sheeprl Reference — `utils.py` (dreamer_v3)

> **Source**: `tmp/sheeprl/sheeprl/algos/dreamer_v3/utils.py` — 235 lines.
> **Purpose** (one-line): Algorithm-specific helpers: `Moments` (return percentile tracker), `compute_lambda_values` (λ-return), `prepare_obs`, `test` (single-episode eval), `AGGREGATOR_KEYS` (metric names), `MODELS_TO_REGISTER` (model-manager registry), weight initializers, and MLflow checkpoint logging.
> **Imports from elsewhere in this index**: [`distribution.md`](distribution.md) (TwoHot encoding produces the `values`/`rewards` distributions whose `.mean` feeds `compute_lambda_values`), [`agent.md`](agent.md) (`PlayerDV3` is the agent driven by `test`; `build_agent` is called by `log_models_from_checkpoint`).

---

## Table of Contents

- [Lines 1–18 — Imports](#lines-118--imports)
- [Lines 20–36 — `AGGREGATOR_KEYS`](#lines-2036--aggregator_keys)
- [Constant (line 37) — `MODELS_TO_REGISTER`](#constant-line-37--models_to_register)
- [Line 40 — `class Moments`](#line-40--class-moments)
- [Line 41 — `Moments.__init__`](#line-41--moments__init__)
- [Line 56 — `Moments.forward`](#line-56--momentsforward)
- [Line 66 — `compute_lambda_values`](#line-66--compute_lambda_values)
- [Line 80 — `prepare_obs`](#line-80--prepare_obs)
- [Line 95 — `test`](#line-95--test)
- [Line 143 — `init_weights`](#line-143--init_weights)
- [Line 170 — `uniform_init_weights`](#line-170--uniform_init_weights)
- [Line 171 — `uniform_init_weights.f` (closure)](#line-171--uniform_init_weightsf-closure)
- [Line 189 — `log_models_from_checkpoint`](#line-189--log_models_from_checkpoint)

---

## Lines 1–18 — Imports

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from torch import Tensor, nn

from sheeprl.utils.env import make_env
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

    from sheeprl.algos.dreamer_v3.agent import PlayerDV3
```

**What this is**: Standard scientific-stack imports plus Lightning `Fabric` (multi-GPU wrapper used to gather percentile statistics across ranks in `Moments.forward`) and sheeprl's `make_env` factory (used inside `test`). The `TYPE_CHECKING` block defers two heavy imports (`mlflow.ModelInfo`, `PlayerDV3`) to type-check time only — preventing circular imports between `utils.py` and `agent.py`.

---

## Lines 20–36 — `AGGREGATOR_KEYS`

```python
AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "State/kl",
    "State/post_entropy",
    "State/prior_entropy",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
}
```

**What this is**: The canonical set of scalar metric names the DreamerV3 training loop registers with sheeprl's `MetricAggregator`. Three families: (1) **Rewards / Game** — environment-side rollout stats. (2) **Loss/** — the six DreamerV3 loss components (world-model total + observation/reward/continue/state heads, plus actor `policy_loss` and critic `value_loss`). (3) **State/** — KL between prior and posterior of the RSSM plus their entropies. (4) **Grads/** — gradient-norm scalars per optimizer. The set is consumed by `dreamer_v3.py` when instantiating the aggregator.

---

## Constant (line 37) — `MODELS_TO_REGISTER`

```python
MODELS_TO_REGISTER = {"world_model", "actor", "critic", "target_critic", "moments"}
```

**What this is**: The set of state-dict keys the sheeprl model-manager / checkpoint system persists for DreamerV3. Matches the keys in the `state` dict passed to `log_models_from_checkpoint` below — `world_model`, `actor`, `critic`, `target_critic` are produced by `build_agent`; `moments` is the return-normalization tracker defined just below.

---

## Line 40 — `class Moments`

```python
class Moments(nn.Module):
```

**What this is**: Running EMA tracker of the low/high percentiles (default 5th and 95th) of imagined returns. Implements the DreamerV3 paper's return-normalization mechanism that makes the actor's entropy coefficient scale-invariant: gradients of the actor objective are divided by `max(1, percentile_high − percentile_low)`. Registered as a buffer-only `nn.Module` so it travels with the model state-dict and is checkpointed via `MODELS_TO_REGISTER`.

---

## Line 41 — `Moments.__init__`

```python
def __init__(
    self,
    decay: float = 0.99,
    max_: float = 1e8,
    percentile_low: float = 0.05,
    percentile_high: float = 0.95,
) -> None:
    super().__init__()
    self._decay = decay
    self._max = torch.tensor(max_)
    self._percentile_low = percentile_low
    self._percentile_high = percentile_high
    self.register_buffer("low", torch.zeros((), dtype=torch.float32))
    self.register_buffer("high", torch.zeros((), dtype=torch.float32))
```

**What this is**: Stores the EMA decay (default 0.99, matching the paper), the percentile pair, and a `_max` ceiling that prevents the inverse-scale from blowing up when `high - low` is tiny. The two scalar EMA states `low` and `high` are registered as buffers (not parameters) so they (a) are saved/loaded with the module and (b) move with `.to(device)` but are not touched by the optimizer.

---

## Line 56 — `Moments.forward`

```python
def forward(self, x: Tensor, fabric: Fabric) -> Any:
    gathered_x = fabric.all_gather(x).float().detach()
    low = torch.quantile(gathered_x, self._percentile_low)
    high = torch.quantile(gathered_x, self._percentile_high)
    self.low = self._decay * self.low + (1 - self._decay) * low
    self.high = self._decay * self.high + (1 - self._decay) * high
    invscale = torch.max(1 / self._max, self.high - self.low)
    return self.low.detach(), invscale.detach()
```

**What this is**: One update step. `fabric.all_gather` pools `x` (typically the flattened imagined λ-return tensor) across DDP ranks so the quantiles are computed over the global batch, not just the local one. The two percentile estimates are EMA-mixed into `self.low` / `self.high`. The returned tuple `(offset, invscale)` is consumed by the actor loss in `dreamer_v3.py` as `normalized = (lambda_values - offset) / invscale`. `.detach()` on both outputs prevents gradients from flowing back through the running stats.

---

## Line 66 — `compute_lambda_values`

```python
def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    lmbda: float = 0.95,
):
    vals = [values[-1:]]
    interm = rewards + continues * values * (1 - lmbda)
    for t in reversed(range(len(continues))):
        vals.append(interm[t] + continues[t] * lmbda * vals[-1])
    ret = torch.cat(list(reversed(vals))[:-1])
    return ret
```

**What this is**: Computes the TD(λ) / GAE-style targets the critic regresses against and the actor's advantage uses. Standard backward recursion: `G_t^λ = r_t + γc_t [ (1−λ) v_{t+1} + λ G_{t+1}^λ ]`, with the terminal value bootstrapped from `values[-1:]`. `continues` carries the discount × episode-continuation mask (so γ is baked in upstream). Inputs are typically the `.mean` of [`TwoHotEncodingDistribution`](distribution.md) outputs from the reward and critic heads. The trailing `[:-1]` drops the bootstrap entry so the returned tensor lines up with the rewards horizon.

---

## Line 80 — `prepare_obs`

```python
def prepare_obs(
    fabric: Fabric, obs: Dict[str, np.ndarray], *, cnn_keys: Sequence[str] = [], num_envs: int = 1, **kwargs
) -> Dict[str, Tensor]:
    torch_obs = {}
    for k, v in obs.items():
        torch_obs[k] = torch.from_numpy(v.copy()).to(fabric.device).float()
        if k in cnn_keys:
            torch_obs[k] = torch_obs[k].view(1, num_envs, -1, *v.shape[-2:]) / 255 - 0.5
        else:
            torch_obs[k] = torch_obs[k].view(1, num_envs, -1)

    return torch_obs
```

**What this is**: Observation-preprocessing helper used by `test` (and indirectly by online rollout code with the same contract). For each key in the dict observation: copy → tensor → move to fabric device → cast to float. Image keys (those listed in `cnn_keys`) are reshaped to `(T=1, B=num_envs, C, H, W)` and rescaled from `[0, 255]` uint8 to `[-0.5, 0.5]` float. Vector keys are flattened to `(1, num_envs, feat)`. The leading time-axis of size 1 matches the RSSM's `(T, B, ...)` input contract.

---

## Line 95 — `test`

```python
@torch.no_grad()
def test(
    player: "PlayerDV3",
    fabric: Fabric,
    cfg: Dict[str, Any],
    log_dir: str,
    test_name: str = "",
    greedy: bool = True,
):
    """Test the model on the environment with the frozen model.

    Args:
        player (PlayerDV3): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
        cfg (DictConfig): the hyper-parameters.
        log_dir (str): the logging directory.
        test_name (str): the name of the test.
            Default to "".
        greedy (bool): whether or not to sample the actions.
            Default to True.
    """
    env: gym.Env = make_env(cfg, cfg.seed, 0, log_dir, "test" + (f"_{test_name}" if test_name != "" else ""))()
    done = False
    cumulative_rew = 0
    obs = env.reset(seed=cfg.seed)[0]
    player.num_envs = 1
    player.init_states()
    while not done:
        # Act greedly through the environment
        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
        real_actions = player.get_actions(
            torch_obs, greedy, {k: v for k, v in torch_obs.items() if k.startswith("mask")}
        )
        if player.actor.is_continuous:
            real_actions = torch.stack(real_actions, -1).cpu().numpy()
        else:
            real_actions = torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()

        # Single environment step
        obs, reward, done, truncated, _ = env.step(real_actions.reshape(env.action_space.shape))
        done = done or truncated or cfg.dry_run
        cumulative_rew += reward
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0 and len(fabric.loggers) > 0:
        fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
```

**What this is**: Single-episode evaluation under `torch.no_grad()`. Builds a fresh env via `make_env` (factory pattern — note the trailing `()`), resets [`PlayerDV3`](agent.md) state (`num_envs = 1`, `init_states()`), then loops: preprocess obs → query `player.get_actions` (with `greedy=True` so the actor uses argmax for discrete, mode for continuous) → convert mask keys → stack actions (continuous keeps logits as last axis, discrete takes `argmax`) → step env → accumulate reward. Logs `Test/cumulative_reward` to fabric's logger (typically WandB / TensorBoard) and closes the env. The `cfg.dry_run` short-circuit forces termination on the first step for smoke-tests.

---

## Line 143 — `init_weights`

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
```

**What this is**: Module-applicable weight initializer (used via `model.apply(init_weights)`). Implements Xavier-fan-avg truncated-normal init: `std = sqrt(1 / fan_avg) / 0.8796…`, where `0.8796…` is the truncated-normal correction factor that re-normalizes std after the `±2σ` truncation so the effective std matches `sqrt(1/fan_avg)`. Linear and Conv2d/ConvTranspose2d branches differ only in how `fan_in`/`fan_out` are computed (kernel area × channels for convs). LayerNorm gets weight=1, bias=0. Biases initialized to zero. Adapted from `NM512/dreamerv3-torch/tools.py#L929` — applied to the standard MLP/CNN blocks of the world model and heads.

---

## Line 170 — `uniform_init_weights`

```python
def uniform_init_weights(given_scale):
```

**What this is**: A **factory** that returns an init function `f` parameterized by `given_scale`. The pattern lets callers stamp a particular variance scale per submodule, e.g. the actor head typically gets a much smaller scale than the encoder. Adapted from `NM512/dreamerv3-torch/tools.py#L957`.

---

## Line 171 — `uniform_init_weights.f` (closure)

```python
def f(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = given_scale / denoms
        limit = np.sqrt(3 * scale)
        nn.init.uniform_(m.weight.data, a=-limit, b=limit)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

return f
```

**What this is**: The inner closure returned by `uniform_init_weights`. Implements Xavier-uniform init scaled by `given_scale`: `limit = sqrt(3 * given_scale / fan_avg)` so weights are drawn from `U(−limit, +limit)` with variance `given_scale / fan_avg`. Biases zeroed. LayerNorm branch identical to `init_weights`. No Conv branch — this initializer is used for the small Linear heads (actor / critic / reward / continue) where DreamerV3 wants a configurable variance scale, with the very small scale of the last layers driving near-zero initial outputs.

---

## Line 189 — `log_models_from_checkpoint`

```python
def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    import mlflow  # noqa

    from sheeprl.algos.dreamer_v3.agent import build_agent

    # Create the models
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    world_model, actor, critic, target_critic = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        env.observation_space,
        state["world_model"],
        state["actor"],
        state["critic"],
        state["target_critic"],
    )
    moments = Moments(
        fabric,
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    moments.load_state_dict(state["moments"])

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["world_model"] = mlflow.pytorch.log_model(unwrap_fabric(world_model), artifact_path="world_model")
        model_info["actor"] = mlflow.pytorch.log_model(unwrap_fabric(actor), artifact_path="actor")
        model_info["critic"] = mlflow.pytorch.log_model(unwrap_fabric(critic), artifact_path="critic")
        model_info["target_critic"] = mlflow.pytorch.log_model(target_critic, artifact_path="target_critic")
        model_info["moments"] = mlflow.pytorch.log_model(moments, artifact_path="moments")
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
```

**What this is**: Hydrates a fully-restored agent from a checkpoint `state` dict and logs every component to MLflow as a separate artifact. Guarded by `_IS_MLFLOW_AVAILABLE` (raises if mlflow not installed). Action-space introspection mirrors `build_agent`'s contract: Box → continuous shape, MultiDiscrete → `nvec`, Discrete → `[n]`. Calls [`build_agent`](agent.md) to rebuild `world_model / actor / critic / target_critic` with their loaded state-dicts, then re-instantiates `Moments` and loads its state. Inside a single nested `mlflow.start_run`, each of the five sub-models is logged via `mlflow.pytorch.log_model` after `unwrap_fabric` strips DDP wrappers (target_critic and moments are not fabric-wrapped). The final `mlflow.log_dict(cfg.to_log, "config.json")` archives the run's resolved config. The keys logged match `MODELS_TO_REGISTER` exactly.
