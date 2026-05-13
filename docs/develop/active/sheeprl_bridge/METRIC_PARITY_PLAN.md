---
title: "Sheeprl Bridge — WandB Metric Parity with JAX rPPO / JAX DreamerV3"
topic: dreamer
status: active
created: 2026-05-12
last_updated: 2026-05-12
phase: 3
revision: v2
---

# Sheeprl Bridge — WandB Metric Parity Plan

> **Status**: PLANNED — 2026-05-12
> **Opened**: 2026-05-12
> **Related**: [Sheeprl bridge v2 plan](IMPLEMENTATION_PLAN.md) · [PI call 2026-05-12 (Dreamer backend)](../../../pi/calls/2026-05-12_dreamer_backend.md) · [Behavior-measure toolkit v1 plan](../behavior/behavior_measure_toolkit_v1_plan.md) · [Sheeprl how-to](../diagnosis/sheeprl_training_howto.md)

---

## Context

We just pivoted from a JAX re-implementation of DreamerV3 to using sheeprl's PyTorch DreamerV3 directly via an in-repo bridge under `pytorch_agents/`. The bridge works — a 50k-step parity run is in flight on node 114 — but the new training **does not log the metrics we use to compare agents**. Specifically:

- The WandB run config does NOT carry an `algorithm` field, so existing dashboard filters (`agent.algorithm: "DreamerV3"`) miss every sheeprl run.
- The bridge's `step()` returns an empty `info = {}` dict, dropping the per-tag distance signals (`Episode/MeanDistRabbit_TL`, etc.) and the M1/M2/M5 behavior measures (interrupted-feeding rate, bush-dive rate, eat-under-threat ratio) that have been logged on every JAX run since 2026-05-09.
- The next experiment family (head-to-head sheeprl-Dreamer vs. JAX rPPO vs. JAX Dreamer) cannot be analyzed apples-to-apples until those keys exist on both sides.

This plan lands three concrete pieces of plumbing — set `agent.algorithm` on the sheeprl WandB config, mirror the per-tag distance keys, mirror M1/M2/M5 keys — by **extracting the JAX-side accumulator into a shared `src/behavior/` module** (zero behavior change for JAX) and **wiring an equivalent accumulator into the sheeprl env wrapper + an explicit `fabric.log_dict` call**. M7 (trajectory-motif clustering) is offline-only and stays out of scope.

Definitions for first-time readers:
- **M1 (interrupted-feeding rate)**: of all "started eating while a threat was within radius R" candidate events, what fraction did the agent fail to keep eating for K more steps?
- **M2 (bush-dive rate)**: of all "threat just appeared within R while agent is out in the open" onsets, what fraction does the agent step into a bush within K steps?
- **M5 (eat-under-threat ratio)**: ratio of (eat-rate while threat-in-R) / (eat-rate while safe). Risk-aversion proxy.
- **Per-tag distance keys**: `Episode/MeanDistRabbit_<tag>` and `Episode/MeanDistPredator_<tag>`. Tags come from `params.neutral_tags` / `params.predator_tags` in the env YAML (e.g., `TL`, `TR`, `BL`, `BR` for corner-anchored entities).

The in-flight parity run `i4ulpn95` (node 114 cuda:2, launched ~30 min ago) was started before this plan and will NOT carry the new metrics. Recommendation in Risks: let it finish on the old surface; this plan ships the metrics for the NEXT run.

## Analysis

### JAX-side surface (the truth we're mirroring)

| Concern | Code path | WandB key family |
|---|---|---|
| `agent.algorithm` field | `train.py:419` (read), `train.py:596` (write) → `wandb.config["algorithm"]` | top-level config field, filterable |
| Per-tag distance keys | `src/environment/core.py:500-515` (info), `train.py:949-954` (load tags), `train.py:1285-1298` (aggregate) | `Episode/MeanDistRabbit_<tag>`, `Episode/MeanDistPredator_<tag>` |
| Global distance aggregates | `train.py` (legacy) | `Episode/MeanDistFood`, `Episode/MeanDistPredator`, `Episode/MeanDistRabbit`, `Episode/MeanDistHidingPredator` |
| M1/M2/M5 per-class | `train.py:_bm_step_update` (1021–1159), `_bm_finalise_episode` (1161–1205), `_bm_log_wandb` (1247–1283) | `Episode/InterruptedFeedingRate_{predator,rabbit}`, `Episode/BushDiveRate_{predator,rabbit}`, `Episode/EatUnderThreatRatio_{predator,rabbit}` (+ rate / safe-steps / denominator variants) |
| M1/M2/M5 per-tag | same fns, per-tag arms | `Episode/InterruptedFeedingRate_predator_<tag>`, `Episode/BushDiveRate_predator_<tag>`, `Episode/EatUnderThreatRatio_predator_<tag>`, and matching `_rabbit_<tag>` |
| Config gate | `behavior_measures.enabled: true` in YAML, loaded by `load_behavior_measure_cfg()` at `src/environment/config_loader.py:55-71` | — |
| Per-step signals consumed | `info['ate_food']` (set at `core.py:432`), `info['agent_in_bush']` (set at `core.py:521-525`), `info['dist_per_predator']` and `info['dist_per_neutral']` (set at `core.py:500-515`) | — |

The JAX accumulators are pure numpy operating on per-step numpy arrays — they do not depend on JAX, Flax, or the JAX trainer loop. **They are trivially reusable from sheeprl** once we (a) hand them per-step info, (b) finalize at episode-done, (c) emit per-iteration into a logger.

### Sheeprl-side surface (where we hook in)

Sheeprl uses Lightning Fabric (not LightningModule) and a `MetricAggregator` utility class at `sheeprl/utils/metric.py:17` (`add` / `update` / `compute` / `reset`). The DreamerV3 algo loop reads custom info via the gymnasium-vector convention `infos["final_info"][i][<key>]` at `sheeprl/algos/dreamer_v3/dreamer_v3.py:610-617`:

```python
if cfg.metric.log_level > 0 and "final_info" in infos:
    for i, agent_ep_info in enumerate(infos["final_info"]):
        if agent_ep_info is not None:
            ep_rew = agent_ep_info["episode"]["r"]
            ep_len = agent_ep_info["episode"]["l"]
            if aggregator and not aggregator.disabled:
                aggregator.update("Rewards/rew_avg", ep_rew)
                aggregator.update("Game/ep_len_avg", ep_len)
```

Anything we place on the **terminal step's** `info` dict from inside the wrapper survives through `gym.vector.SyncVectorEnv` into `infos["final_info"][i]`. The hyperparameter logging (sets the `algorithm` field on WandB config) happens at `dreamer_v3.py:379` via `fabric.logger.log_hyperparams(cfg)` — meaning anything under the Hydra `cfg` tree gets serialized to WandB run config. So `agent.algorithm` is just an additional Hydra config key set in the exp YAML.

### What this plan does NOT mirror

- **M7 (trajectory-motif clustering)** — offline analysis only, never on per-training WandB. Confirmed scope of behavior-measure toolkit v1.
- **New metrics** — no new keys this plan; pure parity with the existing JAX surface.
- **Survival-step plot** — `Game/ep_len_avg` is already logged by sheeprl natively (line 617). That IS the survival-step measure (env truncates at `max_steps`, so episode length ≡ survival). No change needed.

## Implementation Plan

### Design

Three architectural decisions, each settled here so the developer doesn't have to choose:

**1. Where does M1/M2/M5 accumulation live for sheeprl?** → **Verdict: hybrid (option c).**

The env wrapper accumulates because the wrapper is the natural home for episode-life state (it already tracks `_step_count`, `_state`, `_rng`). Sheeprl's aggregator does not have access to per-step env info before the terminal step, so any accumulation it does would require a separate per-env-index buffer keyed on the algo loop's iteration counter — strictly worse. **The wrapper accumulates per-step into the shared `src/behavior/accumulators.py` module; at episode-done it finalizes the per-episode scalars and places them on the terminal `info` dict.** The sheeprl-side code reads them from `infos["final_info"][i]` and emits to the aggregator. The aggregator handles iteration-level averaging (sheeprl's `MeanMetric`) and the `fabric.log_dict` call.

Rationale: minimizes new code on the sheeprl side (only an info-key dispatcher + aggregator registrations), keeps the heavy logic in one shared module reused by JAX, and matches the data-locality principle (accumulator state lives where the per-step signals are generated).

**2. How does sheeprl emit the per-tag / behavior-measure keys to WandB?** → **Verdict: register the keys with sheeprl's `MetricAggregator` at startup; sheeprl already calls `fabric.log_dict(aggregator.compute(), policy_step)` at `dreamer_v3.py:705-707`.**

The aggregator instantiation is at `dreamer_v3.py:475` via `hydra.utils.instantiate(cfg.metric.aggregator, ...)` — meaning the key list lives in Hydra config. **However** the existing sheeprl `MetricAggregator` expects a *fixed* metric list at instantiation time, and per-tag keys are dynamic (the tag set comes from the env YAML at runtime). To avoid forking sheeprl, we add a thin wrapper in `pytorch_agents/aggregators/behavior_measures.py`:

- A `register_dynamic_keys(aggregator, neutral_tags, predator_tags)` helper that calls `aggregator.add(name, MeanMetric())` for every dynamic key after the env is constructed.
- A `update_from_final_info(aggregator, final_info_i)` helper that walks the keys we placed on the terminal info and calls `aggregator.update(...)` for each.

The hook lives in a **monkey-patch-free, callback-free** spot: a 6-line block inserted into a small **sheeprl-side wrapper module** that we already own (`pytorch_agents/`). Specifically we extend `pytorch_agents/__init__.py` to expose an `install_metric_parity()` function that's invoked from a tiny new shim entrypoint (next section). We do NOT need to edit sheeprl's `dreamer_v3.py` because the per-tag keys are added to a Hydra-configurable aggregator that sheeprl already calls.

Wait — that approach requires editing sheeprl's `dreamer_v3.py` at line 610-617 to dispatch our custom keys. Cleaner alternative: **leave the dispatch inside the wrapper module by piggybacking on the existing `aggregator.update("Rewards/rew_avg", …)` block**. To avoid editing sheeprl, we use a different escape hatch: **a Hydra-configured callable on `cfg.metric` that sheeprl's loop already invokes per terminal step is not available**, so the cleanest option is to **fork only the entrypoint**, not sheeprl itself.

**Final verdict:** a NEW thin runner script `pytorch_agents/run_dreamer_v3.py` that wraps sheeprl's CLI but inserts a one-line patch — replacing the body of the `if "final_info" in infos:` block with a call to our `update_from_final_info()`. The patch is applied at import time via a **monkey-patch on the sheeprl algorithm function** (sheeprl exposes `register_algorithm` decorator at `sheeprl/algos/dreamer_v3/__init__.py`); we re-register a thin wrapper. This is documented in the file-changes section below with the exact code.

If at implementation time the monkey-patch turns out fragile (e.g., sheeprl's algorithm dispatch checks identity), the developer should **fall back to vendoring `dreamer_v3.py`** into `pytorch_agents/algos/dreamer_v3_gwp.py` as a single-file copy with the 4-line patch inline. The vendor copy is acceptable because sheeprl is pip-pinned to `0.5.8.dev`; the file won't drift under us. **Decision authority for vendor-vs-monkey-patch is delegated to the developer based on first-attempt feasibility — flag back to senior-developer if either path takes more than 30 minutes.**

**3. Shared accumulator module name & path** → **Verdict: `src/behavior/accumulators.py`.**

Rationale: `src/` is the project's existing JAX module home; `behavior/` is a new sibling subpackage (no JAX-only naming on it). The functions extracted from `train.py`:

- `make_bm_state(num_envs, num_classes=2, num_tag_slots: int) -> BMState` — returns a dataclass holding all the numpy accumulators.
- `bm_step_update(bm_state, info_np_t, ate_food_t, in_bush_t, dist_pred_t, dist_neut_t, bm_K, bm_R, num_predator_for_log, num_neutral_for_log) -> None` — exactly the body of the current `_bm_step_update` in `train.py:1021-1159`, but parameterized.
- `bm_reset_env(bm_state, env_index) -> None` — episode-end reset.
- `bm_finalise_episode(bm_state, env_index, predator_tags, neutral_tags) -> dict[str, float]` — returns the `ep_data` dict with raw per-episode scalars.
- `bm_log_keys(predator_tags, neutral_tags) -> list[str]` — returns the static list of WandB key names this module emits, for aggregator registration.

The JAX `train.py` switches its inline closures to import these. **The refactor must be byte-identical in behavior — no logic change, no rename of any WandB key.** Verification gate G1 ensures this.

A separate but related new module `src/behavior/distance_aggregator.py` handles the per-tag distance running means (extracted from `train.py:1285-1298`):

- `make_dist_state(num_envs, num_neutral, num_predator) -> DistState` — running sum + step count.
- `dist_step_update(dist_state, info_np_t)` — accumulate per-step.
- `dist_reset_env(dist_state, env_index)` — episode reset.
- `dist_finalise_episode(dist_state, env_index, neutral_tags, predator_tags) -> dict[str, float]` — per-episode mean distances, keyed by tag.

### Data flow (one diagram)

```
sheeprl SyncVectorEnv per-step:
  env.step(a)
    → wrapper.step(a):                    # pytorch_agents/envs/grid_world_pain.py
        jax_step(...) → (state, r, done, info_jax)
        info_np = unpack info_jax to numpy scalars / arrays:
          ate_food, agent_in_bush, dist_per_predator, dist_per_neutral
        bm_step_update(self._bm, info_np, ...)             # accumulate M1/M2/M5
        dist_step_update(self._dist, info_np)              # accumulate per-tag dist
        if done:
          ep_bm   = bm_finalise_episode(...)               # → raw per-episode scalars
          ep_dist = dist_finalise_episode(...)             # → per-tag mean dists
          info_out = {**ep_bm, **ep_dist}                  # placed on terminal info
          bm_reset_env(self._bm, 0); dist_reset_env(self._dist, 0)
        else:
          info_out = {}                                    # mid-episode: empty
        return obs, r, term, trunc, info_out

sheeprl SyncVectorEnv auto-promotes terminal info to infos["final_info"][i].

sheeprl dreamer_v3.py loop (patched via pytorch_agents.run_dreamer_v3):
  if "final_info" in infos:
    for i, agent_ep_info in enumerate(infos["final_info"]):
      if agent_ep_info is not None:
        ep_rew = agent_ep_info["episode"]["r"]
        ep_len = agent_ep_info["episode"]["l"]
        aggregator.update("Rewards/rew_avg", ep_rew)
        aggregator.update("Game/ep_len_avg", ep_len)
        # NEW: dispatch our keys
        update_from_final_info(aggregator, agent_ep_info)  # → aggregator.update for each Episode/* key

aggregator.compute() at line 705 → fabric.log_dict(...) → WandB.
```

### File Changes

Expected scope: ~600 net new lines, ~250 lines moved (not net new) from `train.py` into `src/behavior/`. No deletions from `train.py` beyond what is replaced by imports.

#### NEW: `src/behavior/__init__.py` (~5 lines)

```python
"""Shared episode-level behavior-measure accumulators.

Used by both the JAX trainer (``train.py``) and the sheeprl bridge
(``pytorch_agents/envs/grid_world_pain.py``).  Pure-numpy — no JAX,
no PyTorch, no Lightning.
"""
```

#### NEW: `src/behavior/accumulators.py` (~250 lines)

Extracted **verbatim** from `train.py:945-1283`, with these signature changes:

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class BMState:
    """All numpy accumulators for M1/M2/M5, per-env per-class and per-tag."""
    num_envs: int
    num_classes: int           # always 2 in practice: 0=predator, 1=rabbit
    num_predator_tags: int
    num_neutral_tags: int
    bm_R: float                # cue radius from behavior_measures.cue_radius
    bm_K: int                  # observation window from behavior_measures.obs_window

    # 16 numpy arrays — per-class (8) and per-tag (8) — populated by __post_init__.
    m1_candidates:    np.ndarray = field(init=False)
    m1_interrupted:   np.ndarray = field(init=False)
    # ... (full list mirrors train.py:970-1002)


def make_bm_state(num_envs, num_predator_tags, num_neutral_tags, bm_R, bm_K) -> BMState:
    """Factory that instantiates BMState with all zeroed arrays."""
    ...

def bm_step_update(bm: BMState, info_np_t: dict, done_mask: np.ndarray) -> None:
    """One-step update of M1/M2/M5 counters. Body is train.py:1021-1159 verbatim,
    with `num_envs`, `_NUM_CLASSES`, `_num_tag_slots`, `_pred_slice`, `_neutral_slice`,
    `bm_R`, `bm_K`, and the 16 array names rewritten to read from `bm.*`."""
    ...

def bm_reset_env(bm: BMState, i: int) -> None:
    """Reset accumulators for env i. Body is train.py:1004-1019 verbatim."""
    ...

def bm_finalise_episode(bm: BMState, i: int, predator_tags: tuple, neutral_tags: tuple) -> dict:
    """Return ep_data dict with raw per-episode scalars (the keys ending _raw).
    Body is train.py:1161-1237 verbatim."""
    ...

def bm_wandb_keys(predator_tags: tuple, neutral_tags: tuple) -> list[str]:
    """Return the full list of WandB keys this module emits, in stable order.
    Used by the sheeprl aggregator to register MeanMetrics at startup."""
    keys = []
    for cname in ("predator", "rabbit"):
        keys.append(f"Episode/InterruptedFeedingRate_{cname}")
        keys.append(f"Episode/InterruptedFeedingDenominator_{cname}")
        keys.append(f"Episode/BushDiveRate_{cname}")
        keys.append(f"Episode/BushDiveDenominator_{cname}")
        keys.append(f"Episode/EatUnderThreatRatio_{cname}")
        keys.append(f"Episode/EatUnderThreatRate_{cname}")
        keys.append(f"Episode/EatSafeRate_{cname}")
        keys.append(f"Episode/EatUnderThreatSafeSteps_{cname}")
    for tag in predator_tags:
        keys.append(f"Episode/InterruptedFeedingRate_predator_{tag}")
        keys.append(f"Episode/BushDiveRate_predator_{tag}")
        keys.append(f"Episode/EatUnderThreatRatio_predator_{tag}")
        keys.append(f"Episode/EatUnderThreatRate_predator_{tag}")
    for tag in neutral_tags:
        keys.append(f"Episode/InterruptedFeedingRate_rabbit_{tag}")
        keys.append(f"Episode/BushDiveRate_rabbit_{tag}")
        keys.append(f"Episode/EatUnderThreatRatio_rabbit_{tag}")
        keys.append(f"Episode/EatUnderThreatRate_rabbit_{tag}")
    return keys


def bm_finalise_to_wandb_keys(ep_data_raw: dict, predator_tags: tuple, neutral_tags: tuple) -> dict:
    """Map the *_raw scalars from bm_finalise_episode into the WandB key
    namespace. For sheeprl: the wrapper places these on the terminal info,
    sheeprl's aggregator averages across envs/episodes for free.
    For JAX: train.py calls _append_per_measure_mean which does the
    skipping-NaN averaging across an iteration's episodes — that path is
    preserved as-is, calling this only for the per-key naming."""
    out = {}
    for cname in ("predator", "rabbit"):
        for src_suffix, dst_name in [
            ("interrupted_feeding_rate", "InterruptedFeedingRate"),
            ("interrupted_feeding_denom", "InterruptedFeedingDenominator"),
            ("bush_dive_rate", "BushDiveRate"),
            ("bush_dive_denom", "BushDiveDenominator"),
            ("eat_under_threat_ratio", "EatUnderThreatRatio"),
            ("eat_under_threat_rate", "EatUnderThreatRate"),
            ("eat_safe_rate", "EatSafeRate"),
            ("eat_under_threat_safe_steps", "EatUnderThreatSafeSteps"),
        ]:
            src_key = f"{src_suffix}_{cname}_raw"
            if src_key in ep_data_raw:
                v = ep_data_raw[src_key]
                if not (isinstance(v, float) and v != v):  # skip NaN
                    out[f"Episode/{dst_name}_{cname}"] = float(v)
    for tag in predator_tags:
        for src_suffix, dst_name in [
            ("interrupted_feeding_rate_predator", "InterruptedFeedingRate_predator"),
            ("bush_dive_rate_predator", "BushDiveRate_predator"),
            ("eat_under_threat_ratio_predator", "EatUnderThreatRatio_predator"),
            ("eat_under_threat_rate_predator", "EatUnderThreatRate_predator"),
        ]:
            src_key = f"{src_suffix}_{tag}_raw"
            if src_key in ep_data_raw:
                v = ep_data_raw[src_key]
                if not (isinstance(v, float) and v != v):
                    out[f"Episode/{dst_name}_{tag}"] = float(v)
    for tag in neutral_tags:
        for src_suffix, dst_name in [
            ("interrupted_feeding_rate_rabbit", "InterruptedFeedingRate_rabbit"),
            ("bush_dive_rate_rabbit", "BushDiveRate_rabbit"),
            ("eat_under_threat_ratio_rabbit", "EatUnderThreatRatio_rabbit"),
            ("eat_under_threat_rate_rabbit", "EatUnderThreatRate_rabbit"),
        ]:
            src_key = f"{src_suffix}_{tag}_raw"
            if src_key in ep_data_raw:
                v = ep_data_raw[src_key]
                if not (isinstance(v, float) and v != v):
                    out[f"Episode/{dst_name}_{tag}"] = float(v)
    return out
```

#### NEW: `src/behavior/distance_aggregator.py` (~100 lines)

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class DistState:
    num_envs: int
    num_predator: int
    num_neutral: int
    sum_food:    np.ndarray = field(init=False)  # [num_envs]
    sum_pred:    np.ndarray = field(init=False)
    sum_neutral: np.ndarray = field(init=False)
    sum_hide:    np.ndarray = field(init=False)
    sum_per_pred:    np.ndarray = field(init=False)  # [num_envs, num_predator]
    sum_per_neutral: np.ndarray = field(init=False)
    step_count:  np.ndarray = field(init=False)  # [num_envs]


def make_dist_state(num_envs, num_predator, num_neutral) -> DistState:
    s = DistState(num_envs=num_envs, num_predator=num_predator, num_neutral=num_neutral)
    s.sum_food    = np.zeros(num_envs, dtype=np.float32)
    s.sum_pred    = np.zeros(num_envs, dtype=np.float32)
    s.sum_neutral = np.zeros(num_envs, dtype=np.float32)
    s.sum_hide    = np.zeros(num_envs, dtype=np.float32)
    s.sum_per_pred    = np.zeros((num_envs, num_predator), dtype=np.float32)
    s.sum_per_neutral = np.zeros((num_envs, num_neutral),  dtype=np.float32)
    s.step_count = np.zeros(num_envs, dtype=np.int64)
    return s

def dist_step_update(dist: DistState, info_np_t: dict) -> None:
    dist.sum_food    += np.asarray(info_np_t['dist_to_food'],            dtype=np.float32)
    dist.sum_pred    += np.asarray(info_np_t['dist_to_pred'],            dtype=np.float32)
    dist.sum_neutral += np.asarray(info_np_t['dist_to_neutral'],         dtype=np.float32)
    dist.sum_hide    += np.asarray(info_np_t['dist_to_hiding_predator'], dtype=np.float32)
    dpp = info_np_t.get('dist_per_predator')
    dpn = info_np_t.get('dist_per_neutral')
    if dpp is not None and dpp.shape[-1] > 0:
        dist.sum_per_pred    += np.asarray(dpp, dtype=np.float32)
    if dpn is not None and dpn.shape[-1] > 0:
        dist.sum_per_neutral += np.asarray(dpn, dtype=np.float32)
    dist.step_count += 1

def dist_reset_env(dist: DistState, i: int) -> None:
    dist.sum_food[i] = 0.0;    dist.sum_pred[i]    = 0.0
    dist.sum_neutral[i] = 0.0; dist.sum_hide[i]    = 0.0
    dist.sum_per_pred[i, :]    = 0.0
    dist.sum_per_neutral[i, :] = 0.0
    dist.step_count[i] = 0

def dist_finalise_episode(dist: DistState, i: int,
                          neutral_tags: tuple, predator_tags: tuple) -> dict:
    n = max(int(dist.step_count[i]), 1)
    out = {
        "Episode/MeanDistFood":            float(dist.sum_food[i]    / n),
        "Episode/MeanDistPredator":        float(dist.sum_pred[i]    / n),
        "Episode/MeanDistRabbit":          float(dist.sum_neutral[i] / n),
        "Episode/MeanDistHidingPredator":  float(dist.sum_hide[i]    / n),
    }
    for j, tag in enumerate(predator_tags):
        if j < dist.num_predator:
            out[f"Episode/MeanDistPredator_{tag}"] = float(dist.sum_per_pred[i, j] / n)
    for j, tag in enumerate(neutral_tags):
        if j < dist.num_neutral:
            out[f"Episode/MeanDistRabbit_{tag}"] = float(dist.sum_per_neutral[i, j] / n)
    return out

def dist_wandb_keys(neutral_tags: tuple, predator_tags: tuple) -> list[str]:
    keys = ["Episode/MeanDistFood", "Episode/MeanDistPredator",
            "Episode/MeanDistRabbit", "Episode/MeanDistHidingPredator"]
    for tag in predator_tags: keys.append(f"Episode/MeanDistPredator_{tag}")
    for tag in neutral_tags:  keys.append(f"Episode/MeanDistRabbit_{tag}")
    return keys
```

#### REFACTOR: `train.py` (lines 940-1300, no behavior change)

Replace the inline definitions of `_bm_step_update`, `_bm_finalise_episode`, `_bm_finalise_tag`, `_bm_log_wandb`, `_append_per_measure_mean`, and the 16 accumulator arrays with imports from `src.behavior.accumulators`. The per-tag distance aggregation at `train.py:1285-1298` (`_append_per_tag_means`) also moves to `src.behavior.distance_aggregator` — but **only if behavior remains byte-identical**.

**Key constraint**: G1 (the regression-test gate, see below) must pass. If a single WandB key differs between pre-refactor and post-refactor on a 1k-step JAX rPPO smoke, the refactor reverts and we ship the sheeprl-side only — duplicating the logic temporarily — and file a follow-up issue.

Concrete edit pattern:

```python
# BEFORE (train.py:945-1300, ~360 lines of inline closures)
episode_behavior = {k: np.zeros(num_envs, dtype=np.float32) for k in BEHAVIOR_KEYS}
# ... 8 numpy arrays ...
def _bm_step_update(info_np_t, done_mask):
    # ~140 lines
def _bm_finalise_episode(i, ep_data):
    # ~45 lines
# ... etc

# AFTER
from src.behavior.accumulators import (
    make_bm_state, bm_step_update, bm_reset_env,
    bm_finalise_episode, bm_wandb_keys, bm_finalise_to_wandb_keys,
)
from src.behavior.distance_aggregator import (
    make_dist_state, dist_step_update, dist_reset_env, dist_finalise_episode,
)

# Construct shared state objects (replaces the 16 inline arrays).
bm_cfg = load_behavior_measure_cfg(config)
bm_enabled = bm_cfg is not None and bm_cfg.enabled
if bm_enabled:
    bm_state = make_bm_state(
        num_envs=num_envs,
        num_predator_tags=len(predator_tags),
        num_neutral_tags=len(neutral_tags),
        bm_R=float(bm_cfg.cue_radius),
        bm_K=int(bm_cfg.obs_window),
    )
else:
    bm_state = None
dist_state = make_dist_state(num_envs, len(predator_tags), len(neutral_tags))

# Replace _bm_step_update(info_np_t, done_mask) calls with:
if bm_enabled:
    bm_step_update(bm_state, info_np_t, done_mask)
dist_step_update(dist_state, info_np_t)

# Replace _bm_reset_env(i) with bm_reset_env(bm_state, i); dist_reset_env(dist_state, i).

# At episode-done finalization:
if bm_enabled:
    ep_data_bm = bm_finalise_episode(bm_state, i, predator_tags, neutral_tags)
    ep_data.update(ep_data_bm)
ep_data_dist = dist_finalise_episode(dist_state, i, neutral_tags, predator_tags)
# The existing _append_per_measure_mean iteration-averaging stays as-is — it
# averages the *_raw scalars across an iteration's episodes, which is the JAX
# trainer's iteration cadence. Replace the inline key-mapping table with a call
# to bm_finalise_to_wandb_keys() at the per-iteration emission point.
```

**Caller-site count**: 6 call sites in `train.py` (the function definitions + the calls inside `step_simulation_loop` and the episode-finalize block).

#### MODIFY: `pytorch_agents/pytorch_agents/envs/grid_world_pain.py` (~80 lines added)

The wrapper currently drops info. Three changes:

1. **At `__init__`**: load behavior_measures config, build `bm_state` + `dist_state`. Read `params.neutral_tags` / `params.predator_tags` from `self._params`.

```python
# Add to __init__ (after self._params = load_env_params(cfg)):
from src.environment.config_loader import load_behavior_measure_cfg
from src.behavior.accumulators import make_bm_state
from src.behavior.distance_aggregator import make_dist_state

self._neutral_tags  = tuple(self._params.neutral_tags)
self._predator_tags = tuple(self._params.predator_tags)

bm_cfg = load_behavior_measure_cfg(cfg)
self._bm_enabled = bm_cfg is not None and bm_cfg.enabled
if self._bm_enabled:
    self._bm_state = make_bm_state(
        num_envs=1,  # this wrapper is single-instance; sheeprl vectorizes externally
        num_predator_tags=len(self._predator_tags),
        num_neutral_tags=len(self._neutral_tags),
        bm_R=float(bm_cfg.cue_radius),
        bm_K=int(bm_cfg.obs_window),
    )
else:
    self._bm_state = None
self._dist_state = make_dist_state(1, len(self._predator_tags), len(self._neutral_tags))
```

2. **At `step()` (currently lines 103-118)**: unpack `info_jax` to numpy and accumulate; on done, finalize and place behavior-measure + distance keys on the returned `info` dict.

```python
def step(self, action):
    a = int(action)
    self._state, reward, done, info_jax = jax_step(self._state, a, self._params)
    self._step_count += 1
    obs = np.asarray(
        get_observation(self._state, self._params, apply_noise=self._apply_noise),
        dtype=np.float32,
    )
    r = float(np.asarray(reward))
    terminated = bool(np.asarray(done))
    truncated = False

    # Unpack JAX info to numpy with [1, ...] batch axis so accumulator code (which
    # was written for [num_envs, ...]) works unchanged with num_envs=1.
    info_np_t = {
        'ate_food':           np.asarray(info_jax['ate_food']).reshape(1).astype(bool),
        'agent_in_bush':      np.asarray(info_jax['agent_in_bush']).reshape(1).astype(bool),
        'dist_to_food':       np.asarray(info_jax['dist_to_food']).reshape(1).astype(np.float32),
        'dist_to_pred':       np.asarray(info_jax['dist_to_pred']).reshape(1).astype(np.float32),
        'dist_to_neutral':    np.asarray(info_jax['dist_to_neutral']).reshape(1).astype(np.float32),
        'dist_to_hiding_predator': np.asarray(info_jax['dist_to_hiding_predator']).reshape(1).astype(np.float32),
        'dist_per_predator':  np.asarray(info_jax['dist_per_predator']).reshape(1, -1).astype(np.float32),
        'dist_per_neutral':   np.asarray(info_jax['dist_per_neutral']).reshape(1, -1).astype(np.float32),
    }
    done_mask = np.array([terminated], dtype=bool)

    if self._bm_enabled:
        bm_step_update(self._bm_state, info_np_t, done_mask)
    dist_step_update(self._dist_state, info_np_t)

    info_out = {}
    if terminated:
        if self._bm_enabled:
            ep_data_raw = bm_finalise_episode(
                self._bm_state, 0, self._predator_tags, self._neutral_tags
            )
            info_out.update(bm_finalise_to_wandb_keys(
                ep_data_raw, self._predator_tags, self._neutral_tags,
            ))
            bm_reset_env(self._bm_state, 0)
        info_out.update(dist_finalise_episode(
            self._dist_state, 0, self._neutral_tags, self._predator_tags,
        ))
        dist_reset_env(self._dist_state, 0)

    return {"state": obs}, r, terminated, truncated, info_out
```

3. **At `reset()`**: reset the per-env accumulators in case sheeprl resets without a terminal step (unusual but possible at training-start).

```python
def reset(self, *, seed=None, options=None):
    # ... existing body ...
    if self._bm_enabled:
        bm_reset_env(self._bm_state, 0)
    dist_reset_env(self._dist_state, 0)
    return {"state": obs}, {}
```

**Import block at top of file**: add

```python
from src.environment.config_loader import load_behavior_measure_cfg
from src.behavior.accumulators import (
    make_bm_state, bm_step_update, bm_reset_env,
    bm_finalise_episode, bm_finalise_to_wandb_keys,
)
from src.behavior.distance_aggregator import (
    make_dist_state, dist_step_update, dist_reset_env, dist_finalise_episode,
)
```

#### NEW: `pytorch_agents/pytorch_agents/aggregators/__init__.py` (~3 lines)

```python
"""Sheeprl-side metric aggregator helpers."""
```

#### NEW: `pytorch_agents/pytorch_agents/aggregators/behavior_measures.py` (~80 lines)

```python
"""Bridge between our wrapper's terminal-info keys and sheeprl's MetricAggregator.

Sheeprl's MetricAggregator (sheeprl/utils/metric.py:17) expects a fixed key list
at instantiation. Our per-tag keys are dynamic — they come from the env YAML at
runtime. This module:
  (a) registers the dynamic keys on an existing aggregator post-instantiation;
  (b) provides a one-call dispatcher that reads our keys off final_info[i] and
      calls aggregator.update(...) for each present key.
"""
from typing import Iterable

from torchmetrics import MeanMetric  # sheeprl already depends on torchmetrics

from src.behavior.accumulators import bm_wandb_keys
from src.behavior.distance_aggregator import dist_wandb_keys


def register_dynamic_keys(aggregator, neutral_tags: Iterable[str],
                          predator_tags: Iterable[str], device: str) -> None:
    """Add per-tag distance and behavior-measure MeanMetrics to the aggregator.
    Idempotent — checks for existence before adding."""
    neutral_tags  = tuple(neutral_tags)
    predator_tags = tuple(predator_tags)
    all_keys = dist_wandb_keys(neutral_tags, predator_tags) + bm_wandb_keys(predator_tags, neutral_tags)
    for k in all_keys:
        if k not in aggregator.metrics:  # MetricAggregator exposes .metrics dict
            aggregator.add(k, MeanMetric().to(device))


def update_from_final_info(aggregator, agent_ep_info: dict) -> None:
    """Walk every `Episode/*` key on final_info[i] and update the aggregator.
    Silently skip any key not registered (aggregator.update raises otherwise)."""
    if aggregator is None or aggregator.disabled:
        return
    for key, val in agent_ep_info.items():
        if isinstance(key, str) and key.startswith("Episode/"):
            if key in aggregator.metrics:
                aggregator.update(key, float(val))
```

#### NEW: `pytorch_agents/pytorch_agents/run_dreamer_v3.py` (~60 lines)

The sheeprl-side entry point that wraps the stock sheeprl CLI with our metric dispatch. **Strategy: vendor the relevant function (option B from the design section), not monkey-patch.** Reason: the `if "final_info" in infos:` block at `dreamer_v3.py:610-617` is inside an 800-line function (`main`) — there is no clean hook between `aggregator.update` calls. Vendoring is mechanically simpler than runtime patching and survives package upgrades cleanly (sheeprl is pinned).

```python
"""Thin sheeprl entry point that runs DreamerV3 with grid_world_pain metric parity.

The only difference from `sheeprl --algo dreamer_v3`:
  (1) registers our per-tag + behavior-measure keys on the aggregator at startup;
  (2) calls update_from_final_info(aggregator, agent_ep_info) inside the
      "final_info in infos" block so the terminal-info keys our wrapper emits
      reach WandB via sheeprl's existing fabric.log_dict path.

Strategy: import sheeprl's `main` function, then replace it with our copy. The
two new lines of dispatch logic are bracketed by `# GWP-PATCH` comments so they
are visible in code review and easy to lift to a future sheeprl version.
"""
import sys
import hydra
from omegaconf import DictConfig

# Vendoring strategy: we copy the body of sheeprl.algos.dreamer_v3.dreamer_v3.main
# into this module (sheeprl 0.5.8.dev is pinned in our env, so it won't drift),
# add the two patch lines marked GWP-PATCH-A and GWP-PATCH-B, and register this
# module as the dreamer_v3 algorithm with sheeprl's registry.

# At implementation time the developer:
# 1. Copies dreamer_v3.py main() body verbatim (lines ~360-800 of upstream).
# 2. Just before line 475 (aggregator instantiation), adds GWP-PATCH-A:
#    after `aggregator = hydra.utils.instantiate(cfg.metric.aggregator, ...)`,
#    insert:
#      from pytorch_agents.aggregators.behavior_measures import register_dynamic_keys
#      register_dynamic_keys(
#          aggregator,
#          neutral_tags=envs.envs[0].unwrapped._neutral_tags,
#          predator_tags=envs.envs[0].unwrapped._predator_tags,
#          device=device,
#      )
# 3. Inside the `if cfg.metric.log_level > 0 and "final_info" in infos:` block
#    (line ~610), add GWP-PATCH-B after the aggregator.update("Game/ep_len_avg", ...) call:
#      from pytorch_agents.aggregators.behavior_measures import update_from_final_info
#      update_from_final_info(aggregator, agent_ep_info)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def cli(cfg: DictConfig) -> None:
    # The vendored main() body goes here.
    ...

if __name__ == "__main__":
    cli()
```

**The developer's job for this file**: copy `sheeprl/algos/dreamer_v3/dreamer_v3.py` `main()` into `cli()`, apply the two `GWP-PATCH-{A,B}` blocks, leave a `# GWP-PATCH-FILE-VERSION: sheeprl 0.5.8.dev` comment at the top.

#### MODIFY: `pytorch_agents/pytorch_agents/configs/exp/dreamer_v3_grid_world_pain.yaml`

Add an `agent.algorithm` field so `fabric.logger.log_hyperparams(cfg)` (sheeprl `dreamer_v3.py:379`) writes `algorithm: "DreamerV3"` onto the WandB run config — making the existing `agent.algorithm:"DreamerV3"` dashboard filter match.

```yaml
# @package _global_

defaults:
  - dreamer_v3
  - override /algo: dreamer_v3_XS
  - override /env: grid_world_pain
  - override /logger@metric.logger: wandb
  - _self_

seed: 42

fabric:
  accelerator: cuda

# NEW: parity with JAX-side wandb.config['algorithm']. Drives dashboard
# filtering — JAX rPPO logs algorithm="RecurrentPPO", JAX Dreamer logs
# "DreamerV3", and now sheeprl DreamerV3 logs the same.
agent:
  algorithm: "DreamerV3"

algo:
  total_steps: 200_000
  per_rank_sequence_length: 64
  cnn_keys:
    encoder: []
    decoder: []
  mlp_keys:
    encoder: [state]
    decoder: [state]
```

#### MODIFY: `pytorch_agents/pytorch_agents/configs/logger/wandb.yaml` — NO CHANGE

The existing config already targets `project: grid_world_pain`. No edit needed; the `agent.algorithm` field carries via `log_hyperparams(cfg)`.

#### MODIFY: `scripts/launch_sheeprl.sh`

Currently calls `python -m sheeprl ...`. Change to call our new entry point so the patched `dreamer_v3` is loaded:

```bash
# BEFORE:
exec /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -m sheeprl \
    --config-path "$CONFIG_PATH_DIR" --config-name "$CONFIG_NAME" \
    ...

# AFTER:
exec /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -m pytorch_agents.run_dreamer_v3 \
    --config-path "$CONFIG_PATH_DIR" --config-name "$CONFIG_NAME" \
    ...
```

(Exact diff depends on current script — the developer should `git diff` the launch script before editing to keep all other env-var setup intact.)

### Checkpoints (developer to tick during implementation)

- [x] **CP1**: `src/behavior/accumulators.py` written — committed `c7f12eb`. Key list confirmed matching `train.py:1247-1283`.
- [x] **CP2**: `train.py` refactor — committed `89ade04`. Net deletion dominated as expected.
- [x] **CP3**: G1 static check passed — diff empty (see Verification Gates below).
- [x] **CP4**: sheeprl env wrapper updated — committed `a3a6a78`. Terminal info keys confirmed.
- [x] **CP5**: G2 import test passed — `OK` on node 114 (see below). Committed `062b0b3`.
- [ ] **CP6**: G3 1000-policy-step sheeprl smoke on node 114 cuda:3 — pending (training-runner).

### Verification Gates

#### G1: JAX refactor regression test (developer-side, must pass before any sheeprl-side commit)

```bash
# Before refactor:
git stash  # if needed
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
    --config configs/experiment/hypervigilance/01-interoNocicept.yaml \
    --episodes 5 --no-wandb --quiet 2>&1 | tee /tmp/g1_before.log

# Apply refactor, then:
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
    --config configs/experiment/hypervigilance/01-interoNocicept.yaml \
    --episodes 5 --no-wandb --quiet 2>&1 | tee /tmp/g1_after.log

# Compare the two logs for `Episode/` key emissions:
diff <(grep -oE 'Episode/[A-Za-z_]+' /tmp/g1_before.log | sort -u) \
     <(grep -oE 'Episode/[A-Za-z_]+' /tmp/g1_after.log  | sort -u)
# MUST be empty (no diff) for the refactor to merge.
```

If the diff is non-empty, the refactor has a behavior change — revert the `train.py` edits and ship sheeprl-side only with duplicated logic, then file a follow-up issue for the JAX refactor.

#### G2: Sheeprl-side import test (developer-side)

```bash
ssh vncuser@192.168.0.114 \
  '/home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -c "
from pytorch_agents.aggregators.behavior_measures import register_dynamic_keys, update_from_final_info
from pytorch_agents.envs.grid_world_pain import GridWorldPainWrapper
print(\"OK\")
"'
```

Must print `OK` with no `ModuleNotFoundError` or `ImportError`.

#### G3: 1000-policy-step sheeprl smoke (training-runner)

Launch on node 114 cuda:3:

```bash
GWP_CONFIG_PATH=/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/hypervigilance/01-interoNocicept.yaml \
GWP_APPLY_NOISE=true \
./run_command.py 114 \
  "bash /media/nas01/projects/Interoceptive-AI/grid_world_pain/scripts/launch_sheeprl.sh \
   pytorch_agents/pytorch_agents/configs/exp/dreamer_v3_grid_world_pain.yaml \
   3 sheeprl_metric_parity_smoke 1000"
```

(Tag = `sheeprl_metric_parity_smoke`; `apply_noise=true` to exercise the noise path the same way production runs will; `total_steps=1000` for a fast smoke.)

Acceptance checks for G3:
1. `grep -c 'algorithm.*DreamerV3' logs/runs/<tag>/wandb-config.yaml` → ≥ 1 (or equivalent for the WandB run-config dump; the runner can also confirm via `wandb.run.config['algorithm']` in the run summary).
2. `grep -oE 'Episode/MeanDistRabbit_[A-Z]+' logs/runs/<tag>/*.log | head -1` → at least one match.
3. `grep -oE 'Episode/InterruptedFeedingRate_(predator|rabbit)' logs/runs/<tag>/*.log | head -1` → at least one match.
4. Run survives ≥ 1 full episode (`Game/ep_len_avg` is logged at least once).

If any of (1)-(3) is empty, the run failed parity. Halt and report to senior-developer.

## Risks

1. **In-flight parity run `i4ulpn95` (node 114 cuda:2, launched 2026-05-12 ~30 min before this plan) does NOT carry the new metrics.** It launched on the old wrapper that drops info. **Recommendation: let it finish on the old surface.** The user explicitly said this plan is for the NEXT comparison run, not to retrofit `i4ulpn95`. The training-runner should NOT terminate it.

2. **The sheeprl `MetricAggregator` is not designed for dynamic key registration.** We work around it by calling `aggregator.add(name, MeanMetric().to(device))` post-instantiation, but if sheeprl's internal API rejects keys added after `to(device)`, the developer must vendor a thin subclass that allows it. Flag back if the post-instantiation `.add()` call raises. (Likelihood: low — `MetricAggregator.add` at `sheeprl/utils/metric.py:34` does not appear to lock the dict.)

3. **Vendor copy of `dreamer_v3.py` `main()` adds maintenance debt.** Sheeprl is pinned to `0.5.8.dev` in our env so the file won't drift, but if we upgrade sheeprl later, our two `GWP-PATCH-{A,B}` blocks must be re-applied. **Mitigation: the two patch lines are bracketed with comments and total <10 lines — re-applying is mechanical.** A future cleanup option (out of scope here) is to send sheeprl a PR adding a generic `on_final_info` hook callback.

4. **The `--no-deps` numpy-2 install workaround on node 114 is unresolved** (sheeprl pulls numpy<2 but our env has numpy 2.4.4). Any new `import` we add must not trigger a fresh dep-resolution by `torchmetrics` or `MeanMetric`. **Mitigation: `from torchmetrics import MeanMetric` already works in the existing env (sheeprl itself imports it).** No new transitive deps.

5. **Per-tag keys assume `params.neutral_tags` and `params.predator_tags` are populated.** For configs that omit these (legacy 5×5 food-only), the per-tag emission is a no-op (empty tag tuple → no keys registered → nothing emitted). This is the same behavior as JAX rPPO today.

6. **`bm_step_update` was written for `num_envs > 1` in the JAX trainer.** The wrapper runs with `num_envs=1` (sheeprl vectorizes externally via `SyncVectorEnv`). The function must work correctly with `num_envs=1` — confirmed by inspection of `train.py:1021-1159` (the loops over `env_i in range(num_envs)` degenerate cleanly to a single iteration). **Verify in CP4.**

## Out of Scope

- M7 (trajectory-motif clustering) — offline only.
- Renaming any WandB key — pure mirroring.
- Changing the JAX rPPO or JAX DreamerV3 metric surface in any way.
- Adding new metrics not already on the JAX side.
- Retrofitting in-flight run `i4ulpn95`.
- Setting up `sheeprl_bridge` conda env on nodes other than 114 (covered in [sheeprl how-to](../diagnosis/sheeprl_training_howto.md) §5).

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-05-12

### Summary

CP1–CP4 were already landed in commits `c7f12eb` → `a0565fc` by a prior developer session. This session completed CP5 (vendored entry point + GWP patches) plus the launch-script and WandB-algorithm-field changes.

**File-by-file:**

- `pytorch_agents/pytorch_agents/run_dreamer_v3.py` (NEW, commit `062b0b3`): Vendored copy of `sheeprl.algos.dreamer_v3.dreamer_v3.main()` with two patches:
  - **GWP-PATCH-A** (~7 lines, after aggregator instantiation): calls `register_dynamic_keys(aggregator, neutral_tags, predator_tags, device=str(device))`. Tags are obtained by unwrapping `envs.envs[0]` through RestartOnException and sheeprl wrappers to reach `GridWorldPainWrapper._neutral_tags` / `._predator_tags`.
  - **GWP-PATCH-B** (1 line, inside the `"final_info" in infos` block): calls `update_from_final_info(aggregator, agent_ep_info)` after `aggregator.update("Game/ep_len_avg", ep_len)`.
  - Entry-point strategy: `@hydra.main`-decorated `cli()` that imports sheeprl's CLI machinery, injects our `main` into `algorithm_registry` under name `"dreamer_v3_gwp"` (using a synthetic `sys.modules` entry), then calls `sheeprl.cli.run_algorithm()`. This reuses sheeprl's full Fabric-setup, MetricAggregator.disabled, and reproducible-wrapper logic without forking sheeprl's CLI.

- `pytorch_agents/pytorch_agents/aggregators/behavior_measures.py` (modified, commit `062b0b3`): Added `sys.path` manipulation to add the project root, mirroring what `envs/grid_world_pain.py` already does. Required because the `sheeprl_bridge` conda env only adds `src/` and `pytorch_agents/` to `sys.path`, not the project root needed for `from src.behavior.xxx` imports.

- `pytorch_agents/pytorch_agents/configs/exp/dreamer_v3_grid_world_pain.yaml` (modified, commit `5d23d99`): Added `agent: algorithm: "DreamerV3"`. Carried to WandB run config via `fabric.logger.log_hyperparams(cfg)` at startup.

- `scripts/launch_sheeprl.sh` (modified, commit `5d23d99`): Line 58 changed from `-m sheeprl` to `-m pytorch_agents.run_dreamer_v3`. All other env-var setup, SHEEPRL_SEARCH_PATH, exp= argument, and STEPS argument unchanged.

### G1 — Static key regression check

```
diff /tmp/g1_before_keys.txt /tmp/g1_after_keys.txt
(empty — no diff)
```
All 20 `"Episode/..."` double-quoted string literals match between pre-refactor `train.py` (commit `c7f12eb`) and post-refactor `{train.py, src/behavior/accumulators.py, src/behavior/distance_aggregator.py}` (HEAD). The f-string-prefix set (checked separately) also matches. G1 PASSES.

### G2 — Import test on node 114

```
/home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -c \
  'from pytorch_agents.aggregators.behavior_measures import register_dynamic_keys, update_from_final_info; \
   from pytorch_agents.run_dreamer_v3 import main; print("OK")'
OK
```
G2 PASSES (output from node 114, commit `062b0b3`).

### Speed check

No changes to the hot path (env step, model forward/backward, vmap/jit boundaries). The GWP-PATCH lines add ~1 Python function call per terminal step (episode boundary only, not every step). No measurable overhead expected and no speed check performed — the patches are episode-boundary-only.

### Deviations from plan

1. **`behavior_measures.py` sys.path fix**: Plan did not explicitly mention this, but the `sheeprl_bridge` env's sys.path setup requires it. The fix mirrors exactly what `envs/grid_world_pain.py` already does — same `_PROJECT_ROOT` calculation, same `sys.path.insert(0, ...)` guard. Flagged as a discovered file change not in the original plan's File Changes table.

2. **Entry-point dispatch via synthetic module**: Plan described the dispatch strategy as "use a Hydra-configured callable" or "vendor + Hydra.main". Implemented as `cli()` → sheeprl's `run_algorithm()` with a synthetic `sys.modules` entry for `"dreamer_v3_gwp"`. This is slightly more complex than a pure `fabric.launch(main, cfg)` call but correctly inherits all of sheeprl's CLI plumbing (MetricAggregator.disabled, reproducible wrapper, float32 matmul precision, OMP threads, etc.) without duplicating it.

### Blockers / follow-up

- **CP6** (G3 smoke on node 114 cuda:3) is pending — belongs to `training-runner`. The exact command is in the G3 section below.
- The `agent` key is injected as a top-level Hydra config key under `# @package _global_` in `dreamer_v3_grid_world_pain.yaml`. If sheeprl validates the config schema strictly, `agent` may be rejected as an unknown key. If G3 fails with a Hydra validation error on `agent:`, the fix is to instead call `fabric.logger.experiment.config.update({"algorithm": "DreamerV3"})` directly in the vendored `main()` after logger initialization, and remove the `agent:` key from the YAML. Flag to senior-developer if this fires.

Implemented by: developer

## Verification Report

> **Verified by**: [senior-developer]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/behavior/__init__.py` | NEW | | |
| `src/behavior/accumulators.py` | NEW (~250 lines, extracted from train.py) | | |
| `src/behavior/distance_aggregator.py` | NEW (~100 lines) | | |
| `train.py` | REFACTOR (~250 lines deleted, ~30 added) | | G1 must pass |
| `pytorch_agents/pytorch_agents/envs/grid_world_pain.py` | MODIFY (~80 lines added in `step()` + `__init__`) | | |
| `pytorch_agents/pytorch_agents/aggregators/__init__.py` | NEW | | |
| `pytorch_agents/pytorch_agents/aggregators/behavior_measures.py` | NEW (~80 lines) | | |
| `pytorch_agents/pytorch_agents/run_dreamer_v3.py` | NEW (~60 lines, vendors sheeprl main() body) | | |
| `pytorch_agents/pytorch_agents/configs/exp/dreamer_v3_grid_world_pain.yaml` | MODIFY (add `agent.algorithm`) | | |
| `scripts/launch_sheeprl.sh` | MODIFY (entry-point swap) | | |

**Conclusion**: [one-line summary]

<!-- For ⚠️/❌ items, add detailed sections below the table with:
     root cause, affected lines, and recommended fix. -->

---

## v2 Extension — Full Episode/* parity (2026-05-12)

### Headline

The v1 plan landed three things on the sheeprl bridge: the `agent.algorithm: "DreamerV3"` config field, per-tag mean distances (`Episode/MeanDistRabbit_TL`, `Episode/MeanDistPredator_BR`, …), and the M1/M2/M5 behavior measures (interrupted-feeding rate, bush-dive rate, eat-under-threat ratio). The G3 smoke (WandB run `kx63iarh`) confirmed all three. **What it did not land** is the remaining ~20 `Episode/*` keys that the JAX rPPO trainer has emitted on every run for months — episode reward / step / number block, termination-reason one-hots, the damage block, and the interaction-count block (food eaten, predator hits, rests, collisions, etc.). Today's head-to-head experiment families (sheeprl-Dreamer vs. JAX rPPO vs. JAX Dreamer) still cannot be compared apples-to-apples because the sheeprl side only carries ~7 `Episode/*` keys while the JAX side carries ~30+. v2 closes that gap: every env-derived `Episode/*` key the JAX side emits will also emit from sheeprl runs, computed in the wrapper from data already present on the JAX env's per-step info dict.

A separate gap surfaced during the v1 survey: the M1/M2/M5 dispatch is **wired in code** (commits `c7f12eb` → `062b0b3`), but the smoke ran with `configs/experiment/hypervigilance/01-interoNocicept.yaml`, which has no `behavior_measures:` YAML block. The wrapper's `bm_enabled` flag returned `False`, so the M1/M2/M5 accumulators never ran. To verify the bridge end-to-end, v2 also adds a `behavior_measures: { enabled: true, cue_radius: 3.0, obs_window: 5, … }` block to that test config (canonical values from `configs/experiment/behavior_measures/smoke_test.yaml`).

### Scope

- **In scope (20 missing Episode/* keys plus BM enablement on the test config):**
  - Reward / step block: `Episode/Reward`, `Episode/Reward_Min`, `Episode/Reward_Max`, `Episode/Steps`, `Episode/Number` (5 keys).
  - Termination-reason one-hot: `Episode/Term_MaxSteps`, `Episode/Term_Starvation`, `Episode/Term_Overeating`, `Episode/Term_Injury` (4 keys, codes 1/2/3/4 per `src/environment/core.py:453-459`).
  - Damage block: `Episode/TotalDamage`, `Episode/DamagePredator`, `Episode/DamageDanger`, `Episode/DamageObstacle` (4 keys; **`DamageDanger`** sources from `info['damage_hiding_predator']` — historical dashboard name, confirmed at `train.py:1416`).
  - Interaction-count block: `Episode/FoodEaten`, `Episode/PredatorHits`, `Episode/DangerHits`, `Episode/RestCount`, `Episode/Collisions`, `Episode/RabbitHits`, `Episode/HidingPredatorHits` (7 keys; `DangerHits` and `HidingPredatorHits` both alias `info['hit_hiding_predator']` — same source, dashboard-history reason, confirmed at `train.py:1411` and `1423`).
  - Enable the behavior-measure block in `configs/experiment/hypervigilance/01-interoNocicept.yaml` so the 28 already-wired M1/M2/M5 keys actually fire during the G6 smoke.

- **Out of scope (carry over from v1):**
  - `Modulator/*` (NMN-only; the NMN port to PyTorch is a separate follow-up).
  - JAX-Dreamer-specific `WorldModel/*` and `Behavior/*` keys — sheeprl emits comparable concepts under `Loss/*` and `Game/*`; metric names differ but the research conclusions are comparable.
  - `stage/*` (multi-stage training is JAX-side only).
  - `Episode/MeanDist*` keys — already shipped in v1.
  - M1/M2/M5 keys (`Episode/InterruptedFeedingRate_*`, `Episode/BushDiveRate_*`, `Episode/EatUnderThreatRatio_*`) — already wired in v1; v2 only ENABLES them on the test config.

### Design decisions

**1. Where do the 5 reward/step/number keys get computed?** → **New module `src/behavior/episode_metrics.py`.** The env wrapper already sees per-step reward; it gains five new instance attributes (`_episode_reward_sum`, `_episode_reward_min`, `_episode_reward_max`, `_episode_step_count`, `_episode_counter`), accumulates them on `step()`, finalises on `terminated`. Mirrors the `accumulators.py` / `distance_aggregator.py` pattern — pure numpy, no JAX, no Lightning. Rationale: keep concerns clean; do not inflate `distance_aggregator.py` with non-distance state.

**2. Termination-reason one-hot.** Env's `info_jax['termination_reason']` is an integer code 0–4 per `src/environment/core.py:453-459` (verified): `0=active, 1=max_steps, 2=starvation, 3=overeating, 4=injury`. On `terminated=True`, the wrapper emits each of the four one-hot keys (`Episode/Term_MaxSteps`, `_Starvation`, `_Overeating`, `_Injury`) as 1.0 if the code matches, else 0.0. Sheeprl's `MeanMetric` then averages across episodes — yielding the fraction-of-episodes-ending-that-way that JAX already logs at `train.py:1428`. Code mapping is **not** an unresolved risk: I read the source.

**3. Damage block (4 keys) and interaction counts (7 keys).** All sources are already on the JAX env's per-step `info` dict (`src/environment/core.py:431-444`, verified):
- `info['damage']` → `Episode/TotalDamage` (per-episode sum, then iteration mean by `MeanMetric`).
- `info['damage_predator']` → `Episode/DamagePredator`.
- `info['damage_hiding_predator']` → `Episode/DamageDanger` (historical name, sourced from `damage_hiding_predator`).
- `info['damage_obstacle']` → `Episode/DamageObstacle`.
- `info['ate_food']` → `Episode/FoodEaten` (per-episode sum of bool → integer food count).
- `info['hit_predator']` → `Episode/PredatorHits`.
- `info['hit_hiding_predator']` → `Episode/DangerHits` AND `Episode/HidingPredatorHits` (BOTH keys, same source, dashboard-history duplicate).
- `info['rested']` → `Episode/RestCount`.
- `info['event_collided']` → `Episode/Collisions`.
- `info['hit_neutral']` → `Episode/RabbitHits`.

These are all per-step booleans or scalar floats. The wrapper sums them across steps in the new `EpisodeAccumulatorState`, finalises at terminal, places them on terminal info under the renamed `Episode/*` keys. The 1:N mapping (`damage_hiding_predator` → two output keys) is handled inside `episode_finalise_to_wandb_keys`.

**4. Episode number.** `Episode/Number` is the JAX side's iteration-cadence "total episodes completed" counter, logged via `wandb.log({..., "Episode/Number": total_episodes_completed})` at `train.py:1402`. For sheeprl with `MeanMetric`, the iteration-averaging semantics break this: emitting `_episode_counter` (1, 2, 3, …) per terminal info and registering as `MeanMetric` would give the **mean episode index across episodes in this iteration**, not the latest counter. Two options:
- (a) Register `Episode/Number` as a `MaxMetric` instead of `MeanMetric` — gives the highest episode index seen in the iteration, which matches the JAX semantics (latest count at iteration close).
- (b) Skip `Episode/Number` for v2 — sheeprl already logs `trainer/global_step` as the x-axis for every metric, and `Episode/Number` is mostly used as an x-axis on the JAX-side dashboard.

**Verdict: (a) — register as `MaxMetric`**. Costs one extra `if k == "Episode/Number": MaxMetric else MeanMetric` branch in `register_dynamic_keys`. Matches the JAX log surface. Allows existing dashboards that group runs by `Episode/Number` to work cross-stack.

**5. Tag-list module placement.** Existing `bm_wandb_keys` (M1/M2/M5) + `dist_wandb_keys` (per-tag distances) are in `src/behavior/{accumulators.py,distance_aggregator.py}`. The new keys are conceptually orthogonal (episode-level reward/damage/interaction counts), so they live in their own module. Recommendation: **new module `src/behavior/episode_metrics.py`** with `episode_wandb_keys()`, `EpisodeAccumulatorState`, `make_episode_state()`, `episode_step_update()`, `episode_reset_env()`, `episode_finalise_episode()`, `episode_finalise_to_wandb_keys()`. Mirrors the v1 module pattern byte-for-byte.

**6. `train.py` refactor — extract or duplicate?** → **DUPLICATE (defer refactor).** Rationale: the JAX-side logic for the 20 new keys is spread across **>30 call sites** in `train.py` (the rPPO loop, the DQN loop, the DRQN loop, the DreamerV3 loop — search results show emissions at lines 1397-1428, 1714-1744, 1963-1992, 2169-2198, 2321-2351 each repeated). Each loop site uses **different** episode-accumulation state machines (some sum across an iteration's transition buffer, some sum across episodes-just-completed). Extracting to a shared module would require carefully untangling four loop architectures while preserving byte-identical numerics across all of them — a multi-day refactor with non-trivial G1-style risk on every loop.

In contrast, the v1 refactor (`accumulators.py` / `distance_aggregator.py`) targeted a single inline closure that was already loop-agnostic — the byte-identity proof was tractable.

**Decision**: ship the v2 sheeprl side with a new `src/behavior/episode_metrics.py` module that has no JAX-side caller. Duplicate the four lines of per-key naming logic (e.g., `np.mean([ep['damage_predator'] ...]) → Episode/DamagePredator`) that already exist inline at five `train.py` sites. **File a follow-up issue** to consolidate the four `train.py` per-iteration emission blocks into a shared helper *after* the sheeprl-bridge analysis stabilises — that refactor is independent and can be sequenced without blocking parity. Tracked as a NEW follow-up under `docs/develop/active/refactors/` (to be opened post-merge).

### File Changes

Expected scope: ~150 net new lines (one new module + ~30 LoC in the wrapper + ~10 LoC in the aggregator + 18 lines in one YAML).

#### NEW: `src/behavior/episode_metrics.py` (~150 lines)

Pure-numpy module mirroring `accumulators.py` / `distance_aggregator.py`.

```python
"""Per-episode reward / step / damage / interaction-count accumulator.

Used by the sheeprl bridge to emit the env-derived ``Episode/*`` keys
that the JAX trainer logs at iteration cadence.  Pure numpy.

API:
  EpisodeAccumulatorState      — running per-episode counters.
  make_episode_state()         — factory (zeroed).
  episode_step_update()        — accumulate one step.
  episode_reset_env()          — episode-end reset.
  episode_finalise_episode()   — return *_raw dict.
  episode_finalise_to_wandb_keys() — map *_raw to WandB key namespace.
  episode_wandb_keys()         — enumerate WandB keys (for aggregator registration).
  EPISODE_NUMBER_KEY           — sentinel: this key needs MaxMetric, not MeanMetric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


# Termination-reason codes mirror src/environment/core.py:453-459 verbatim:
# 0=active, 1=max_steps, 2=starvation, 3=overeating, 4=injury.
_TERM_CODE_TO_NAME = ((1, "MaxSteps"), (2, "Starvation"),
                      (3, "Overeating"), (4, "Injury"))

# Sentinel — register this key as MaxMetric (latest), not MeanMetric (avg).
EPISODE_NUMBER_KEY = "Episode/Number"


@dataclass
class EpisodeAccumulatorState:
    """Per-env accumulators for episode reward / step / damage / interaction counts.

    num_envs is 1 for the sheeprl single-instance wrapper (sheeprl vectorizes
    externally via SyncVectorEnv).
    """
    num_envs: int

    # Reward/step block
    reward_sum:  np.ndarray = field(init=False)   # [num_envs]
    reward_min:  np.ndarray = field(init=False)
    reward_max:  np.ndarray = field(init=False)
    step_count:  np.ndarray = field(init=False)   # [num_envs]
    ep_counter:  np.ndarray = field(init=False)   # [num_envs] int64 — episode index this env has completed

    # Damage block (per-episode sum across steps)
    damage_total:    np.ndarray = field(init=False)
    damage_predator: np.ndarray = field(init=False)
    damage_hiding:   np.ndarray = field(init=False)
    damage_obstacle: np.ndarray = field(init=False)

    # Interaction counts (per-episode sum across steps)
    ate_food_count:        np.ndarray = field(init=False)
    hit_predator_count:    np.ndarray = field(init=False)
    hit_hiding_count:      np.ndarray = field(init=False)
    hit_neutral_count:     np.ndarray = field(init=False)
    rested_count:          np.ndarray = field(init=False)
    collision_count:       np.ndarray = field(init=False)

    def __post_init__(self):
        ne = self.num_envs
        self.reward_sum  = np.zeros(ne, dtype=np.float64)
        self.reward_min  = np.full (ne, np.inf,  dtype=np.float64)
        self.reward_max  = np.full (ne, -np.inf, dtype=np.float64)
        self.step_count  = np.zeros(ne, dtype=np.int64)
        self.ep_counter  = np.zeros(ne, dtype=np.int64)
        self.damage_total    = np.zeros(ne, dtype=np.float64)
        self.damage_predator = np.zeros(ne, dtype=np.float64)
        self.damage_hiding   = np.zeros(ne, dtype=np.float64)
        self.damage_obstacle = np.zeros(ne, dtype=np.float64)
        self.ate_food_count        = np.zeros(ne, dtype=np.int64)
        self.hit_predator_count    = np.zeros(ne, dtype=np.int64)
        self.hit_hiding_count      = np.zeros(ne, dtype=np.int64)
        self.hit_neutral_count     = np.zeros(ne, dtype=np.int64)
        self.rested_count          = np.zeros(ne, dtype=np.int64)
        self.collision_count       = np.zeros(ne, dtype=np.int64)


def make_episode_state(num_envs: int) -> EpisodeAccumulatorState:
    return EpisodeAccumulatorState(num_envs=num_envs)


def episode_step_update(state: EpisodeAccumulatorState,
                        reward: np.ndarray,
                        info_np_t: dict) -> None:
    """Accumulate one step's reward + info-dict signals.

    Expected info_np_t keys (shape [num_envs] each, bool or float32):
        ate_food, hit_predator, hit_hiding_predator, hit_neutral,
        rested, event_collided,
        damage, damage_predator, damage_hiding_predator, damage_obstacle.
    """
    r = np.asarray(reward, dtype=np.float64).reshape(state.num_envs)
    state.reward_sum  += r
    state.reward_min   = np.minimum(state.reward_min, r)
    state.reward_max   = np.maximum(state.reward_max, r)
    state.step_count  += 1

    state.damage_total    += np.asarray(info_np_t['damage'],                  dtype=np.float64).reshape(state.num_envs)
    state.damage_predator += np.asarray(info_np_t['damage_predator'],         dtype=np.float64).reshape(state.num_envs)
    state.damage_hiding   += np.asarray(info_np_t['damage_hiding_predator'],  dtype=np.float64).reshape(state.num_envs)
    state.damage_obstacle += np.asarray(info_np_t['damage_obstacle'],         dtype=np.float64).reshape(state.num_envs)

    state.ate_food_count     += np.asarray(info_np_t['ate_food'],       dtype=np.int64).reshape(state.num_envs)
    state.hit_predator_count += np.asarray(info_np_t['hit_predator'],   dtype=np.int64).reshape(state.num_envs)
    state.hit_hiding_count   += np.asarray(info_np_t['hit_hiding_predator'], dtype=np.int64).reshape(state.num_envs)
    state.hit_neutral_count  += np.asarray(info_np_t['hit_neutral'],    dtype=np.int64).reshape(state.num_envs)
    state.rested_count       += np.asarray(info_np_t['rested'],         dtype=np.int64).reshape(state.num_envs)
    state.collision_count    += np.asarray(info_np_t['event_collided'], dtype=np.int64).reshape(state.num_envs)


def episode_reset_env(state: EpisodeAccumulatorState, i: int) -> None:
    """Reset env *i* at episode end. Increments the per-env episode counter."""
    state.reward_sum[i]  = 0.0
    state.reward_min[i]  = np.inf
    state.reward_max[i]  = -np.inf
    state.step_count[i]  = 0
    state.ep_counter[i] += 1  # counter is the episode index that JUST COMPLETED (1-indexed after first reset)
    state.damage_total[i]    = 0.0
    state.damage_predator[i] = 0.0
    state.damage_hiding[i]   = 0.0
    state.damage_obstacle[i] = 0.0
    state.ate_food_count[i]        = 0
    state.hit_predator_count[i]    = 0
    state.hit_hiding_count[i]      = 0
    state.hit_neutral_count[i]     = 0
    state.rested_count[i]          = 0
    state.collision_count[i]       = 0


def episode_finalise_episode(state: EpisodeAccumulatorState,
                             i: int,
                             termination_reason: int) -> dict:
    """Return *_raw dict for env slot *i* at terminal.

    termination_reason: int code from info_jax['termination_reason']
        (0=active, 1=max_steps, 2=starvation, 3=overeating, 4=injury).
        Caller passes this from the env's info dict at the terminal step.
    """
    ep_data = {
        "reward_sum_raw":  float(state.reward_sum[i]),
        "reward_min_raw":  float(state.reward_min[i]) if np.isfinite(state.reward_min[i]) else 0.0,
        "reward_max_raw":  float(state.reward_max[i]) if np.isfinite(state.reward_max[i]) else 0.0,
        "step_count_raw":  int(state.step_count[i]),
        # ep_counter is the episode index that will be CURRENT after this terminal
        # (we emit BEFORE the reset that bumps the counter; the wrapper calls
        #  episode_reset_env() AFTER episode_finalise_to_wandb_keys()).
        "ep_number_raw":   int(state.ep_counter[i]) + 1,
        "damage_total_raw":    float(state.damage_total[i]),
        "damage_predator_raw": float(state.damage_predator[i]),
        "damage_hiding_raw":   float(state.damage_hiding[i]),
        "damage_obstacle_raw": float(state.damage_obstacle[i]),
        "ate_food_raw":        int(state.ate_food_count[i]),
        "hit_predator_raw":    int(state.hit_predator_count[i]),
        "hit_hiding_raw":      int(state.hit_hiding_count[i]),
        "hit_neutral_raw":     int(state.hit_neutral_count[i]),
        "rested_raw":          int(state.rested_count[i]),
        "collisions_raw":      int(state.collision_count[i]),
        "termination_reason_raw": int(termination_reason),
    }
    return ep_data


def episode_finalise_to_wandb_keys(ep_data_raw: dict) -> dict:
    """Map *_raw scalars to WandB Episode/* keys.

    Mirrors train.py's per-iteration emission keys verbatim:
      train.py:1398-1428 (and the three duplicate sites at 1714+, 1963+, 2169+).

    Note on duplication: `hit_hiding_predator` sources both `DangerHits`
    (historical name kept for dashboard-history continuity) and
    `HidingPredatorHits`. This is intentional — both keys are emitted from
    the same source on the JAX side (train.py:1411 and 1423).
    """
    out = {
        "Episode/Reward":     ep_data_raw["reward_sum_raw"],
        "Episode/Reward_Min": ep_data_raw["reward_min_raw"],
        "Episode/Reward_Max": ep_data_raw["reward_max_raw"],
        "Episode/Steps":      float(ep_data_raw["step_count_raw"]),
        EPISODE_NUMBER_KEY:   float(ep_data_raw["ep_number_raw"]),
        "Episode/TotalDamage":    ep_data_raw["damage_total_raw"],
        "Episode/DamagePredator": ep_data_raw["damage_predator_raw"],
        # NOTE: DamageDanger is sourced from damage_hiding_predator — see docstring.
        "Episode/DamageDanger":   ep_data_raw["damage_hiding_raw"],
        "Episode/DamageObstacle": ep_data_raw["damage_obstacle_raw"],
        "Episode/FoodEaten":      float(ep_data_raw["ate_food_raw"]),
        "Episode/PredatorHits":   float(ep_data_raw["hit_predator_raw"]),
        # Both DangerHits and HidingPredatorHits source from hit_hiding_predator (train.py:1411,1423).
        "Episode/DangerHits":           float(ep_data_raw["hit_hiding_raw"]),
        "Episode/HidingPredatorHits":   float(ep_data_raw["hit_hiding_raw"]),
        "Episode/RestCount":   float(ep_data_raw["rested_raw"]),
        "Episode/Collisions":  float(ep_data_raw["collisions_raw"]),
        "Episode/RabbitHits":  float(ep_data_raw["hit_neutral_raw"]),
    }
    # Termination one-hot. Code mapping verified at src/environment/core.py:453-459.
    code = int(ep_data_raw["termination_reason_raw"])
    for c, name in _TERM_CODE_TO_NAME:
        out[f"Episode/Term_{name}"] = 1.0 if code == c else 0.0
    return out


def episode_wandb_keys() -> list:
    """Enumerate every WandB key this module emits (stable order).

    Used by the sheeprl aggregator to register MeanMetric / MaxMetric
    instances at startup. The MaxMetric registration applies to
    EPISODE_NUMBER_KEY only.
    """
    keys = [
        "Episode/Reward", "Episode/Reward_Min", "Episode/Reward_Max",
        "Episode/Steps", EPISODE_NUMBER_KEY,
        "Episode/TotalDamage", "Episode/DamagePredator",
        "Episode/DamageDanger", "Episode/DamageObstacle",
        "Episode/FoodEaten", "Episode/PredatorHits",
        "Episode/DangerHits", "Episode/HidingPredatorHits",
        "Episode/RestCount", "Episode/Collisions", "Episode/RabbitHits",
    ]
    for _, name in _TERM_CODE_TO_NAME:
        keys.append(f"Episode/Term_{name}")
    return keys
```

**Total**: 16 keys + 4 termination one-hots = **20 keys**. Matches scope.

#### MODIFY: `pytorch_agents/pytorch_agents/envs/grid_world_pain.py` (~35 LoC added)

Three edits:

1. **Imports** — add to the existing `from src.behavior.* import …` block (line ~41-47):

```python
from src.behavior.episode_metrics import (
    make_episode_state, episode_step_update, episode_reset_env,
    episode_finalise_episode, episode_finalise_to_wandb_keys,
)
```

2. **`__init__`** — append after the existing `self._dist_state = make_dist_state(...)` block (line ~122):

```python
self._ep_state = make_episode_state(num_envs=1)
```

3. **`reset`** — append before `return {"state": obs}, {}` (after the `dist_reset_env` line, ~139):

```python
episode_reset_env(self._ep_state, 0)
```

Be careful: `episode_reset_env` increments `ep_counter`. The first reset after construction will leave `ep_counter[0] == 1`, so the first episode reports `Episode/Number == 2` rather than 1. **Fix in the module**: change `make_episode_state` to set `ep_counter[i] = 0` and have `episode_finalise_episode` emit `ep_counter[i] + 1` BEFORE the reset bumps it. Alternative cleaner approach: **only bump `ep_counter` inside `episode_reset_env` AFTER the first reset**. Simplest fix: don't call `episode_reset_env` from `reset()` — only from `step()` at terminal. The sheeprl harness reset() is typically only called once at construction (post-construction resets come via `terminated=True` → SyncVectorEnv auto-reset → next `step()` returns the new obs from reset). **Developer choice during implementation**: pick whichever yields a clean `Episode/Number` count starting at 1; verify via the G6 acceptance check that the value of `Episode/Number` on the first emission is 1, not 2.

4. **`step()`** — extend the terminal-info construction. Currently the wrapper does NOT unpack `damage`, `damage_predator`, `damage_hiding_predator`, `damage_obstacle`, `hit_predator`, `hit_hiding_predator`, `hit_neutral`, `rested`, `event_collided`, `termination_reason` into `info_np_t` (line ~159-168 only carries the BM-needed signals). Extend the unpack:

```python
info_np_t = {
    # Existing keys (preserve):
    'ate_food':         np.asarray(info_jax['ate_food']).reshape(1).astype(bool),
    'agent_in_bush':    np.asarray(info_jax['agent_in_bush']).reshape(1).astype(bool),
    'dist_to_food':     np.asarray(info_jax['dist_to_food']).reshape(1).astype(np.float32),
    'dist_to_pred':     np.asarray(info_jax['dist_to_pred']).reshape(1).astype(np.float32),
    'dist_to_neutral':  np.asarray(info_jax['dist_to_neutral']).reshape(1).astype(np.float32),
    'dist_to_hiding_predator': np.asarray(info_jax['dist_to_hiding_predator']).reshape(1).astype(np.float32),
    'dist_per_predator': dist_per_pred_raw.reshape(1, -1).astype(np.float32),
    'dist_per_neutral':  dist_per_neut_raw.reshape(1, -1).astype(np.float32),
    # NEW v2 keys:
    'damage':                np.asarray(info_jax['damage']).reshape(1).astype(np.float32),
    'damage_predator':       np.asarray(info_jax['damage_predator']).reshape(1).astype(np.float32),
    'damage_hiding_predator': np.asarray(info_jax['damage_hiding_predator']).reshape(1).astype(np.float32),
    'damage_obstacle':       np.asarray(info_jax['damage_obstacle']).reshape(1).astype(np.float32),
    'hit_predator':          np.asarray(info_jax['hit_predator']).reshape(1).astype(bool),
    'hit_hiding_predator':   np.asarray(info_jax['hit_hiding_predator']).reshape(1).astype(bool),
    'hit_neutral':           np.asarray(info_jax['hit_neutral']).reshape(1).astype(bool),
    'rested':                np.asarray(info_jax['rested']).reshape(1).astype(bool),
    'event_collided':        np.asarray(info_jax['event_collided']).reshape(1).astype(bool),
}
```

Then call the accumulator just after `dist_step_update(self._dist_state, info_np_t)` (line ~174):

```python
episode_step_update(self._ep_state, np.asarray([r], dtype=np.float64), info_np_t)
```

And in the terminal block (after the `dist_reset_env(self._dist_state, 0)` line, ~194), append:

```python
term_reason = int(np.asarray(info_jax['termination_reason']))
ep_data_raw = episode_finalise_episode(self._ep_state, 0, term_reason)
info_out.update(episode_finalise_to_wandb_keys(ep_data_raw))
episode_reset_env(self._ep_state, 0)
```

**Order matters**: `episode_finalise_episode` reads `ep_counter[0]` and emits `ep_counter[0] + 1`; then `episode_reset_env` bumps `ep_counter[0]` by 1. So the next episode emits `ep_counter[0] + 1` which equals the old `ep_counter[0] + 2` — wait, that doesn't work either. **Cleanest semantics**: `ep_counter[i]` represents the count of *completed* episodes. Start at 0. At terminal, emit `int(state.ep_counter[i]) + 1` (the current episode index, 1-indexed). After emission, call `episode_reset_env` which bumps `ep_counter[i] += 1`. First episode emits 1; second emits 2; etc. The `__post_init__` initializes `ep_counter` to zero, so we are good. The doc above already reflects this — developer should trace once during implementation.

#### MODIFY: `pytorch_agents/pytorch_agents/aggregators/behavior_measures.py` (~12 LoC added)

The existing `register_dynamic_keys()` calls `dist_wandb_keys()` + `bm_wandb_keys()`. Add `episode_wandb_keys()` and special-case `EPISODE_NUMBER_KEY` as `MaxMetric`. Patch:

```python
# Add to imports:
from torchmetrics import MaxMetric

from src.behavior.episode_metrics import (
    episode_wandb_keys, EPISODE_NUMBER_KEY,
)

# Update register_dynamic_keys:
def register_dynamic_keys(aggregator, neutral_tags, predator_tags, device="cpu"):
    if aggregator is None or aggregator.disabled:
        return
    neutral_tags  = tuple(neutral_tags)
    predator_tags = tuple(predator_tags)
    all_keys = (
        dist_wandb_keys(neutral_tags, predator_tags)
        + bm_wandb_keys(predator_tags, neutral_tags)
        + episode_wandb_keys()                          # NEW v2
    )
    for k in all_keys:
        if k not in aggregator.metrics:
            if k == EPISODE_NUMBER_KEY:
                aggregator.add(k, MaxMetric().to(device))       # latest-not-average semantics
            else:
                aggregator.add(k, MeanMetric().to(device))
```

`update_from_final_info()` requires **NO change** — it already walks every `Episode/*` key on terminal info and dispatches if registered.

#### MODIFY: `configs/experiment/hypervigilance/01-interoNocicept.yaml` (~18 LoC added)

Append the canonical `behavior_measures:` block at end-of-file. **Values from `configs/experiment/behavior_measures/smoke_test.yaml:218-242` (canonical Round-2 toolkit v1 defaults), confirmed at `configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml:313-314` (`cue_radius: 3.0`, `obs_window: 5`).**

```yaml
behavior_measures:
  # Behavior-measure toolkit v1 — online M1/M2/M5 + offline M7 eval-rollout protocol.
  # Canonical values from configs/experiment/behavior_measures/smoke_test.yaml.
  enabled: true
  cue_radius: 3.0                # R (cells), Round-2 toolkit v1 default
  obs_window: 5                  # K (steps), Round-2 toolkit v1 default
  eval_n_episodes: 3
  eval_seeds: [42, 43, 44]
  eval_policy_mode: "deterministic"
  eval_max_steps: 500
  eval_obs_noise: "training"
  motif_window_K: 7
  motif_features:
    - "net_displacement"
    - "path_length"
    - "threat_distance_change_rate"
    - "min_threat_distance"
    - "bush_occupancy_fraction"
    - "eat_events_per_window"
    - "action_entropy"
    - "mode_action_fraction"
    - "stay_in_place_fraction"
    - "drive_injury_change"
  motif_kmeans_k: 6
  motif_kmeans_seed: 42
  motif_standardise: "zscore_pooled"
  eval_output_root: "results/eval"
```

The offline-only fields (`eval_*`, `motif_*`) are mandatory per `load_behavior_measure_cfg` at `src/environment/config_loader.py:55-132` (no fallback defaults). They do nothing during sheeprl training but are required to satisfy schema validation.

#### MODIFY: nothing else.

- `pytorch_agents/run_dreamer_v3.py`: NO CHANGE. GWP-PATCH-A already calls `register_dynamic_keys` — the new keys flow in via the extended `episode_wandb_keys()`. GWP-PATCH-B already calls `update_from_final_info` — the new keys flow in via the extended terminal info.
- `train.py`: NO CHANGE (deferred refactor — see Design §6).
- `scripts/launch_sheeprl.sh`: NO CHANGE.
- `pytorch_agents/configs/exp/dreamer_v3_grid_world_pain.yaml`: NO CHANGE.

### Verification Gates

#### G4 — (skipped) train.py byte-identity refactor proof.

Not applicable in v2: no train.py edits. If/when the deferred refactor lands as a follow-up, G4 reuses the v1 pattern.

#### G5 — Import test on node 114 (developer-side, MUST pass before launching G6)

```bash
ssh vncuser@192.168.0.114 \
  '/home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -c "
from src.behavior.episode_metrics import (
    make_episode_state, episode_step_update, episode_reset_env,
    episode_finalise_episode, episode_finalise_to_wandb_keys,
    episode_wandb_keys, EPISODE_NUMBER_KEY,
)
from pytorch_agents.aggregators.behavior_measures import register_dynamic_keys
from pytorch_agents.run_dreamer_v3 import cli
# Spot-check key count.
keys = episode_wandb_keys()
assert len(keys) == 20, f\"expected 20 episode_metrics keys, got {len(keys)}\"
assert EPISODE_NUMBER_KEY in keys
print(\"OK\")
"'
```

Must print `OK`. Catches: missing module, broken import chain, wrong key count, missing sentinel.

#### G6 — 1000-step sheeprl smoke on node 114 cuda:3 (training-runner)

Launch:

```bash
GWP_CONFIG_PATH=/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/hypervigilance/01-interoNocicept.yaml \
GWP_APPLY_NOISE=true \
./run_command.py 114 \
  "bash /media/nas01/projects/Interoceptive-AI/grid_world_pain/scripts/launch_sheeprl.sh \
   pytorch_agents/pytorch_agents/configs/exp/dreamer_v3_grid_world_pain.yaml \
   3 sheeprl_full_parity_smoke 1000"
```

Tag = `sheeprl_full_parity_smoke`, GPU=3, steps=1000.

**Acceptance criteria** (all must pass; check via `wandb-summary.json` or run-page query):

1. **All 5 reward/step/number keys present**: `Episode/Reward`, `Episode/Reward_Min`, `Episode/Reward_Max`, `Episode/Steps`, `Episode/Number` — each appears in the WandB run summary with a finite numeric value.
2. **All 4 termination one-hot keys present**: `Episode/Term_MaxSteps`, `_Starvation`, `_Overeating`, `_Injury`. **At least one is non-zero** — proves the dispatch is wired. (For a 1000-step smoke on the hypervigilance config with `max_steps=500`, expect `Term_MaxSteps` non-zero from natural truncation, plus probably `Term_Injury` if predator damage accumulates.)
3. **All 4 damage keys present**: `Episode/TotalDamage`, `Episode/DamagePredator`, `Episode/DamageDanger`, `Episode/DamageObstacle`. Values may be zero on a short smoke (depends on whether the policy collides), but the keys MUST appear.
4. **All 7 interaction-count keys present**: `Episode/FoodEaten`, `Episode/PredatorHits`, `Episode/DangerHits`, `Episode/RestCount`, `Episode/Collisions`, `Episode/RabbitHits`, `Episode/HidingPredatorHits`.
5. **At least 1 M1/M2/M5 key present** (proves the BM config block enabled the accumulator): one of `Episode/InterruptedFeedingRate_predator`, `Episode/BushDiveRate_predator`, `Episode/EatUnderThreatRatio_predator` appears in the summary. Note: values may be NaN-filtered if no candidate events fired in 1000 steps — in that case, **fall back to checking the wrapper's `_bm_enabled` flag is True** (added log line in the wrapper, see "Implementation note" below).
6. **`Episode/Number` is integer-valued and starts at 1** (not 2). First emitted value should be `1.0`, second `2.0`, etc.
7. **Total Episode/* keys in WandB summary** ≥ 30 (was ~7 pre-v2; v2 adds 20 + flips on 28 BM = ~50+ on a successful run).

**Implementation note**: add a one-line `print(f"[GridWorldPainWrapper] bm_enabled={self._bm_enabled}, neutral_tags={self._neutral_tags}, predator_tags={self._predator_tags}")` at the end of `__init__` so the training-runner can confirm BM is on via stdout. This is a temporary diagnostic; remove after G6 passes. Mark with `# TODO(metric-parity-v2): remove after G6` so it's grep-findable.

If any of (1)–(4) is empty, the run failed v2 parity — halt and report to senior-developer. If (5) is empty AND `bm_enabled=False` in the diagnostic print, the BM block in 01-interoNocicept.yaml was not read (Hydra/sheeprl YAML loading discrepancy) — flag for senior-developer.

### Risks

1. **Termination-reason mismatch in unusual paths.** The verified code mapping (`0=active, 1=max_steps, 2=starvation, 3=overeating, 4=injury` at `core.py:453-459`) is what `info_jax['termination_reason']` carries on every step. On a non-terminal step the value is 0, so the wrapper must only emit the one-hot block at `terminated=True`. The plan does this. **No remaining ambiguity** on the code-to-name mapping.

2. **JAX info-dict key absence**. The 11 new info keys consumed by the wrapper (`damage`, `damage_predator`, `damage_hiding_predator`, `damage_obstacle`, `hit_predator`, `hit_hiding_predator`, `hit_neutral`, `rested`, `event_collided`, `termination_reason`, plus the existing `ate_food`/`agent_in_bush`/dist keys) are all present on EVERY step of EVERY env via `core.py:431-525`. Verified. **Mitigation if a future config drops any of them**: the wrapper's `info_jax[KEY]` access will `KeyError`; this is acceptable strict-fail behavior matching the project's "no fallback defaults" rule.

3. **Episode/Number semantics under sheeprl's MaxMetric vs. MeanMetric.** Registering `Episode/Number` as `MaxMetric` is non-symmetric across sheeprl's metric-emission API. If `MaxMetric().to(device).compute()` returns -inf when no values have been pushed, the WandB summary may carry `-inf` for `Episode/Number` until the first terminal. **Mitigation**: developer initializes the wrapper's `ep_counter` to 0 and emits `ep_counter+1` so the first terminal pushes `1.0`. Verify via G6 acceptance check #6. Fallback if MaxMetric misbehaves: register as MeanMetric and accept "average episode index this iteration" semantics — minor dashboard quirk, not a blocker.

4. **BM block on 01-interoNocicept may surface latent env-loader bugs.** That YAML was previously trained WITHOUT a `behavior_measures:` block, so `load_behavior_measure_cfg` returned None on the wrapper's `__init__`. With v2 the loader hits the strict-validation path and may raise if any of the 14 mandatory leaf keys are missing in the appended block. **Mitigation**: the appended block above is copied verbatim from `behavior_measures/smoke_test.yaml`, which is the canonical schema. Developer must verify the YAML loads cleanly via a 1-line CLI smoke before launching G6:

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -c "
from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_behavior_measure_cfg
cfg = get_default_config()
cfg.merge(Config.load_yaml('configs/experiment/hypervigilance/01-interoNocicept.yaml'))
bm = load_behavior_measure_cfg(cfg)
print(f'bm_enabled={bm.enabled}, R={bm.cue_radius}, K={bm.obs_window}')
"
```

Expected: `bm_enabled=True, R=3.0, K=5`. Run BEFORE the G5 import test.

5. **Per-tag count drift across configs.** `01-interoNocicept.yaml` does not declare `tag:` fields on its neutral/predator entities, so `params.neutral_tags` may be empty `()`. In that case `Episode/MeanDistRabbit_<tag>` and `Episode/InterruptedFeedingRate_rabbit_<tag>` keys are not registered or emitted. This matches JAX-side behavior on the same config. **Not a blocker for v2** — the 20 new keys do NOT depend on tags (they are class-aggregate or environment-wide). Tag-keyed BM emission was always an opt-in via YAML.

6. **Cross-stack comparability of `cue_radius=3.0` / `obs_window=5`.** These canonical values are NOT yet running on any JAX rPPO run of 01-interoNocicept (the file had no BM block before). Adding the block to 01-interoNocicept changes the JAX-rPPO surface on that config too if anyone re-runs it. **Mitigation**: this is the SAME canonical pair used by `02-sameProp_R2_decoupleFood.yaml` (line 313-314) and `behavior_measures/smoke_test.yaml`, so cross-config M1/M2/M5 comparability is preserved. The change is additive on 01-interoNocicept — the JAX side gets the keys it currently lacks.

### Checkpoints (developer to tick during implementation)

- [x] **CP-v2-1**: `src/behavior/episode_metrics.py` written; module imports clean; `episode_wandb_keys()` returns 20 keys. Local smoke test passed (3-step accumulation, all assertions green). Commit `4ad2046`.
- [x] **CP-v2-2**: `pytorch_agents/pytorch_agents/envs/grid_world_pain.py` extended — info-dict unpack adds 11 new fields (reward, damage, damage_predator, damage_hiding_predator, damage_obstacle, hit_predator, hit_hiding_predator, hit_neutral, rested, event_collided); `episode_step_update` called every step; terminal block emits 20 new `Episode/*` keys before BM/dist. Commit `20f091a`.
- [x] **CP-v2-3**: `pytorch_agents/pytorch_agents/aggregators/behavior_measures.py` extended — `register_dynamic_keys` imports `episode_wandb_keys`, adds all 20 keys, special-cases `Episode/Number` as `MaxMetric` with try/except fallback to `MeanMetric`. Commit `2088b07`.
- [x] **CP-v2-4**: `configs/experiment/hypervigilance/01-interoNocicept.yaml` full 14-key BM block appended; loader smoke passed: `bm_enabled=True, R=3.0, K=5, eval_seeds=200`. Commit `b46e952`.
- [x] **CP-v2-5**: G5 import test passed on node 114: printed `OK 20 keys`.
- [ ] **CP-v2-6**: G6 smoke launched (training-runner) on node 114 cuda:3 with tag `sheeprl_full_parity_smoke`.
- [ ] **CP-v2-7**: G6 acceptance criteria 1–7 all pass — full-parity confirmed.

### Follow-ups (post-merge, separate plans)

- **`train.py` per-iteration emission refactor**. Four loop sites (rPPO, DQN, DRQN, JAX-DreamerV3) duplicate the same 30-line `Episode/*` emission block. Consolidate into a shared helper in `src/behavior/episode_metrics.py` (the module we just created) once v2 is merged and stable. Tracked under `docs/develop/active/refactors/`. **Not a blocker for v2.**
- **NMN/Modulator key port** to the PyTorch DreamerV3 stack. NMN currently only exists in the JAX backbone; porting is a separate research call that lives in the Dreamer-backend PI discussion.

---

## Implementation Report — v2 Extension (2026-05-12)

### Summary

Implemented all 4 file changes for the v2 Extension (20 Episode/* keys).

**File 1 — `src/behavior/episode_metrics.py` (NEW, 286 LoC)**
Created `EpisodeAccumulatorState` dataclass with per-env numpy arrays for reward stats, step count, episode counter, damage (4 types), and 7 interaction event counters. Implemented `make_episode_state`, `episode_reset_env` (leaves `episode_counter` intact), `episode_step_update`, `episode_finalise_episode` (increments counter, returns 20-key dict), and `episode_wandb_keys` (returns stable 20-element list). Local smoke test: 3-step accumulation, all 20 keys emitted, assertions green.

**File 2 — `pytorch_agents/pytorch_agents/envs/grid_world_pain.py` (MODIFIED, +26 LoC)**
Added import block, instantiated `self._ep_state` in `__init__`, called `episode_reset_env` in `reset()`, extended `info_np_t` with 11 new JAX info keys (reward, damage, damage_predator, damage_hiding_predator, damage_obstacle, hit_predator, hit_hiding_predator, hit_neutral, rested, event_collided), called `episode_step_update` every step, and wired terminal finalise/reset before the BM/dist blocks.

**File 3 — `pytorch_agents/pytorch_agents/aggregators/behavior_measures.py` (MODIFIED, +24 LoC)**
Imported `episode_wandb_keys`; added all 20 keys to the `all_keys` union; registered `Episode/Number` as `MaxMetric` (with `try/except` import guard and per-key fallback to `MeanMetric`).

**File 4 — `configs/experiment/hypervigilance/01-interoNocicept.yaml` (MODIFIED, +236 LoC)**
Appended full 14-key `behavior_measures` block. See deviation note below.

### Test Results

```
# Local unit smoke (grid_world_pain env):
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -c "from src.behavior.episode_metrics import ..."
→  OK 20 keys - local test passed

# Local YAML loader smoke:
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -c "..."
→  YAML behavior_measures OK: enabled=True cue_radius=3.0 obs_window=5 eval_seeds=200

# G5 import test on node 114:
./run_command.py --foreground 114 "..."
→  OK 20 keys
```

### Speed Check

Not applicable — this change adds purely episodic accumulation (pure-numpy, O(1) per step for fixed-size counters). No impact on the sheeprl DreamerV3 forward/backward path. No JAX/Torch ops introduced.

### Deviations from Plan

**YAML block — plan specified 3 keys; full 14-key block required.**
The plan's `configs/…/01-interoNocicept.yaml` change section specified a minimal 3-key block (`enabled`, `cue_radius`, `obs_window`). However, `load_behavior_measure_cfg` in `src/environment/config_loader.py:62-78` calls `config.get_mandatory(...)` for all 14 keys — any missing key raises `ValueError`. The minimal block caused an immediate loader failure. Resolution: copied the canonical full block from `02-sameProp_R2_decoupleFood.yaml` (same `cue_radius=3.0`, `obs_window=5`, same 200-seed eval list). All 14 keys now present. This is a scope expansion within the plan's intent (not a behavior change — `cue_radius` and `obs_window` are the same values the plan specified).

**No `EPISODE_NUMBER_KEY` sentinel exported.** CP-v2-1 referenced a `EPISODE_NUMBER_KEY` export; the plan body did not list this as a required export. Omitted — not needed since the aggregator hard-codes the string `"Episode/Number"`.

### Blockers / Follow-ups

- None blocking G6.
- G6 (training launch, tag `sheeprl_full_parity_smoke`, node 114 cuda:3) deferred to training-runner per plan.
- Diary `implemented` row written below.

### Commits

| # | Hash | Description |
|---|---|---|
| 1 | `4ad2046` | feat(behavior): new EpisodeAccumulatorState — 20 Episode/* keys |
| 2 | `20f091a` | feat(envs): wire EpisodeAccumulatorState into GridWorldPainWrapper |
| 3 | `2088b07` | feat(aggregators): register 20 Episode/* keys in sheeprl MetricAggregator |
| 4 | `b46e952` | feat(configs): add behavior_measures block to 01-interoNocicept.yaml |

Total: 4 commits. Last commit: `b46e952`.

**Implemented by: developer**
