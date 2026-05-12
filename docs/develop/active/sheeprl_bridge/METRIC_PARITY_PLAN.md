---
title: "Sheeprl Bridge — WandB Metric Parity with JAX rPPO / JAX DreamerV3"
topic: dreamer
status: active
created: 2026-05-12
last_updated: 2026-05-12
phase: 3
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
