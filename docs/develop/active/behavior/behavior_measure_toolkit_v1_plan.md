---
title: "Behavior-measure toolkit v1 — implementation plan (M1, M2, M5, M7)"
topic: behavior
status: implemented
created: 2026-05-11
last_updated: 2026-05-11
phase: 1
verification_status: pending
---

# Behavior-measure toolkit v1 — implementation plan (M1, M2, M5, M7)

> **Status**: PLANNED
> **Opened**: 2026-05-11
> **Related**:
> - Design doc (the source of truth for *what* is being built): [`docs/experiments/active/behavior_measures/behavior_measure_toolkit_v1_design.md`](../../../experiments/active/behavior_measures/behavior_measure_toolkit_v1_design.md).
> - Idea memo (background, biological grounding): [`docs/project/ideas/20260510_behavior_measure_toolkit.md`](../../../project/ideas/20260510_behavior_measure_toolkit.md).
> - Architectural precedent — per-tag distance metric (the 5-site `train.py` pattern this plan extends): [`per_quadrant_and_per_rabbit_logging.md`](../hypervigilance/per_quadrant_and_per_rabbit_logging.md).
> - Synthetic-test-trap lesson (load-bearing for verification): [`.claude-memory/memories/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md`](../../../../.claude-memory/memories/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md).
> - Frontmatter / topic contract: [`FRONTMATTER_CONTRACT.md`](../meta/FRONTMATTER_CONTRACT.md). (`topic` is restricted to the existing enum; this plan uses `topic: behavior`. The user's verbal request placed the doc under `docs/develop/active/behavior_measures/`; this plan files it under `behavior/` instead — see "Open questions" item 1.)

---

## 0. Plain-English entry point

The project measures behaviour in two ways today: episode survival steps, and class-mean distances (predator vs. rabbit, plus the just-shipped per-tag fan-out per labelled instance). These are episode-mean statistics — they collapse the temporal structure of what the agent actually does on the few steps that matter. The Round-2.5 sameProp verdict made this concrete: the agent looked like it was avoiding predators on aggregate, but per-tag distances showed it was just camping the safe corner; what mean distance could not show was whether the agent **stopped eating** when a predator approached, **dove into a bush**, or behaved the same on its 3rd encounter as on its 1st. The behaviour-measure toolkit v1 ships four measures from the postdoc's 8-candidate menu that fill those gaps: M1 interrupted-feeding rate, M2 bush-dive rate, M5 eat-under-threat ratio (all three online, emitted at episode-end during training), and M7 defensive-motif repertoire (offline, computed by clustering threat-window trajectories from a deterministic eval-rollout). The design doc has already locked the protocol: R = 3.0 cells, K = 5 steps, K_motif = 7 steps, N = 200 deterministic eval episodes, k = 6 motif clusters, 10 handcrafted features. This plan turns that protocol into a file-by-file implementation surface that `developer` can execute. The eight-step ordering, file-by-file change shape, T1–T8 test specifications, and the verification protocol that catches the dict-assembly trap from the last release are spelled out below. The first downstream user is the v1 demonstration on the saved Round-2.5 Cell A1 (`nm8gn7y2`) and Cell C (`bdnfc0lu`) checkpoints — both run offline, so the v1 demo is gated on the eval-rollout + motif scripts, not on new trainings.

---

## 1. Context

This plan implements four behavioural measures already specified in detail by the experiment-designer (design doc §§1–7). The design fixes the parameters; this plan fixes the *files, lines, and tests*. The toolkit extends the same 5-site accumulator pattern in `train.py` that the per-tag distance metric ships under — that precedent has a `verification_status: pass` and a documented bug-class (the "synthetic-test trap" insight) that motivates the verification protocol below. Two scripts are new: `scripts/eval_rollout.py` (loads a frozen checkpoint, runs 200 deterministic episodes, dumps per-episode `.npz` + a `threat_onsets.parquet` window-index) and `scripts/motif_cluster.py` (reads the dumps, computes 10 handcrafted features, k-means clusters into 6 motifs). The env hook is one boolean per step (`info['agent_in_bush']`), surfaced by lifting an already-computed value out of the predator state-machine block in `core.py`. The schema loader gains a `behavior_measures:` block with 14 mandatory keys (no fallback defaults, per project rule).

---

## 2. Analysis — file-by-file change map

For each file: function / line range, change shape, risk surface. All file paths are repo-relative.

### 2.1 `src/environment/state.py`

**No new `EnvParams` fields required.** The bush mask `params.obs_hides_agent` already exists on `EnvParams` (used in `core.py:152` by the predator state machine). The new measure config (`behavior_measures.cue_radius`, `obs_window`, …) does **not** belong on `EnvParams` because (a) it never enters the JIT graph — it is consumed only by host-side numpy accumulators in `train.py` and by the offline scripts; (b) putting it on `EnvParams` would force a recompile every time `obs_window` changes. The measures consume the config object directly, not `EnvParams`.

**Risk surface**: none for this file. Skip.

### 2.2 `src/environment/config_loader.py`

**Function**: `load_env_params(config)` (or wherever `EnvParams` is constructed; module level). The new `behavior_measures:` block does NOT load into `EnvParams`. Instead, add a thin module-level loader that returns a plain Python dataclass / namedtuple (or just a validated `dict`) consumed by `train.py` and the new scripts. Recommended shape — a frozen dataclass `BehaviorMeasureCfg` at the top of `config_loader.py`, alongside `_normalise_tag`. Every field uses `config.get_mandatory(...)`; no `dict.get(..., default)` for any of the 14 keys.

```python
# In src/environment/config_loader.py, near _normalise_tag.

from dataclasses import dataclass, field
from typing import Tuple

@dataclass(frozen=True)
class BehaviorMeasureCfg:
    enabled: bool
    cue_radius: float
    obs_window: int
    eval_n_episodes: int
    eval_seeds: Tuple[int, ...]
    eval_policy_mode: str            # "deterministic" | "stochastic"
    eval_max_steps: int
    eval_obs_noise: str              # "training" | "zero" | "custom"
    motif_window_K: int
    motif_features: Tuple[str, ...]
    motif_kmeans_k: int
    motif_kmeans_seed: int
    motif_standardise: str           # "zscore_pooled" | "zscore_per_agent" | "none"
    eval_output_root: str


_ALLOWED_POLICY_MODES = {"deterministic", "stochastic"}
_ALLOWED_NOISE_MODES = {"training", "zero", "custom"}
_ALLOWED_STANDARDISE = {"zscore_pooled", "zscore_per_agent", "none"}
_DEFAULT_FEATURE_NAMES = (
    "net_displacement", "path_length", "threat_distance_change_rate",
    "min_threat_distance", "bush_occupancy_fraction", "eat_events_per_window",
    "action_entropy", "mode_action_fraction", "stay_in_place_fraction",
    "drive_injury_change",
)


def load_behavior_measure_cfg(config) -> "BehaviorMeasureCfg | None":
    """Load behavior_measures: from the YAML config.

    Returns None if the top-level key is absent (backwards compatibility — existing
    configs that pre-date this feature load unchanged).  If the block IS present,
    every leaf key is mandatory and missing keys raise ValueError.
    """
    if config.get("behavior_measures") is None:
        return None  # backwards-compat: feature off, online accumulators no-op.

    enabled         = config.get_mandatory("behavior_measures.enabled")
    cue_radius      = config.get_mandatory("behavior_measures.cue_radius", float)
    obs_window      = config.get_mandatory("behavior_measures.obs_window", int)
    eval_n_eps      = config.get_mandatory("behavior_measures.eval_n_episodes", int)
    eval_seeds_raw  = config.get_mandatory("behavior_measures.eval_seeds")
    eval_pol_mode   = config.get_mandatory("behavior_measures.eval_policy_mode")
    eval_max_steps  = config.get_mandatory("behavior_measures.eval_max_steps", int)
    eval_obs_noise  = config.get_mandatory("behavior_measures.eval_obs_noise")
    motif_K         = config.get_mandatory("behavior_measures.motif_window_K", int)
    motif_features  = config.get_mandatory("behavior_measures.motif_features")
    motif_k         = config.get_mandatory("behavior_measures.motif_kmeans_k", int)
    motif_seed      = config.get_mandatory("behavior_measures.motif_kmeans_seed", int)
    motif_std       = config.get_mandatory("behavior_measures.motif_standardise")
    eval_output     = config.get_mandatory("behavior_measures.eval_output_root")

    # ----- validation -----
    if cue_radius <= 0:
        raise ValueError(f"behavior_measures.cue_radius must be > 0; got {cue_radius}.")
    if obs_window < 1:
        raise ValueError(f"behavior_measures.obs_window must be >= 1; got {obs_window}.")
    if eval_n_eps < 1:
        raise ValueError(f"behavior_measures.eval_n_episodes must be >= 1; got {eval_n_eps}.")
    if not isinstance(eval_seeds_raw, (list, tuple)):
        raise ValueError(f"behavior_measures.eval_seeds must be a list/tuple; got {type(eval_seeds_raw)}.")
    if len(eval_seeds_raw) != eval_n_eps:
        raise ValueError(
            f"behavior_measures.eval_seeds length ({len(eval_seeds_raw)}) != eval_n_episodes ({eval_n_eps})."
        )
    eval_seeds = tuple(int(s) for s in eval_seeds_raw)
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("behavior_measures.eval_seeds contains duplicates.")
    if eval_pol_mode not in _ALLOWED_POLICY_MODES:
        raise ValueError(f"behavior_measures.eval_policy_mode must be in {_ALLOWED_POLICY_MODES}; got {eval_pol_mode!r}.")
    if eval_max_steps < 1:
        raise ValueError(f"behavior_measures.eval_max_steps must be >= 1; got {eval_max_steps}.")
    if eval_obs_noise not in _ALLOWED_NOISE_MODES:
        raise ValueError(f"behavior_measures.eval_obs_noise must be in {_ALLOWED_NOISE_MODES}; got {eval_obs_noise!r}.")
    if motif_K < 1:
        raise ValueError(f"behavior_measures.motif_window_K must be >= 1; got {motif_K}.")
    if not motif_features or not all(isinstance(f, str) for f in motif_features):
        raise ValueError("behavior_measures.motif_features must be a non-empty list of strings.")
    unknown_features = set(motif_features) - set(_DEFAULT_FEATURE_NAMES)
    if unknown_features:
        raise ValueError(
            f"behavior_measures.motif_features contains unknown names: {sorted(unknown_features)}. "
            f"Allowed v1 features: {_DEFAULT_FEATURE_NAMES}."
        )
    if motif_k < 2:
        raise ValueError(f"behavior_measures.motif_kmeans_k must be >= 2; got {motif_k}.")
    if motif_std not in _ALLOWED_STANDARDISE:
        raise ValueError(f"behavior_measures.motif_standardise must be in {_ALLOWED_STANDARDISE}; got {motif_std!r}.")

    return BehaviorMeasureCfg(
        enabled=bool(enabled),
        cue_radius=float(cue_radius),
        obs_window=int(obs_window),
        eval_n_episodes=int(eval_n_eps),
        eval_seeds=eval_seeds,
        eval_policy_mode=str(eval_pol_mode),
        eval_max_steps=int(eval_max_steps),
        eval_obs_noise=str(eval_obs_noise),
        motif_window_K=int(motif_K),
        motif_features=tuple(str(f) for f in motif_features),
        motif_kmeans_k=int(motif_k),
        motif_kmeans_seed=int(motif_seed),
        motif_standardise=str(motif_std),
        eval_output_root=str(eval_output),
    )
```

**Backwards compat**: configs that do NOT have a `behavior_measures:` block return `None` and the online accumulators are no-ops (see §2.6). Adding `behavior_measures.enabled: false` is the explicit way to disable online emission while keeping the schema present.

**Risk surface**: forgetting any `get_mandatory` would silently default; the long unit test T1 exercises each missing-key path.

### 2.3 `src/environment/core.py`

**Function**: `jax_step` (`src/environment/core.py:289-…`). The bush-occupancy boolean already exists at line 150 as a local variable inside `update_predators` (called from inside `jax_step`). The cleanest pattern is to **recompute** it in the `info`-dict region of `jax_step` (lines 432–516) — small extra cost (one `jnp.any(jnp.logical_and(...))`), no scope-passing across the helper boundary, and matches the existing `dist_per_*` pattern. Insert after line 515 (`info['dist_per_predator'] = dist_per_predator`):

```python
# In src/environment/core.py jax_step, after the dist_per_* assignments (line ~515):

# Bush occupancy: True iff agent is standing on an obstacle marked hides_agent.
# Mirrors the agent_hidden computation inside update_predators (line 150);
# recomputed here at minimal cost because EnvParams is in scope and we want it on `info`.
agent_in_bush = jnp.any(jnp.logical_and(
    jnp.all(state.obs_pos == new_agent_pos, axis=-1),
    params.obs_hides_agent
))
info['agent_in_bush'] = agent_in_bush
```

`state.obs_pos` is the obstacle position array (already in scope at line 152 of `update_predators`, threaded through `state` here). If `state.obs_pos.shape[0] == 0` the `jnp.any` returns False (empty `axis=0` reduction), which is the correct semantic (no obstacles → never in a bush).

**Risk surface**:
- `new_agent_pos` is the post-step position (the position the agent moved into this step). That is the correct one for M2's "agent dives into bush" semantics. Using `state.agent_pos` (pre-step) would be wrong — log it as a developer checkpoint.
- vmap/JIT safety: `state.obs_pos` and `params.obs_hides_agent` have fixed static shape `[num_obs, 2]` and `[num_obs]`; the reduction is bounded by config; no recompile risk.
- Backwards compatibility: this adds one new key to the `info` dict. All callers (RPPO `StepInfo`, Dreamer transition dict, the existing accumulator sites) explicitly enumerate the keys they read, so a new key is purely additive (mirrors the per-tag pattern).

### 2.4 `src/models/recurrent_ppo_trainer.py`

**Function**: `StepInfo` NamedTuple (`src/models/recurrent_ppo_trainer.py:7-23`, currently 17 fields after the per-tag work). Add **one** new field at the end:

```python
class StepInfo(NamedTuple):
    """Per-step environment info carried through scan for behavioral logging."""
    ate_food: jnp.ndarray
    ...
    termination_reason: jnp.ndarray
    dist_per_neutral: jnp.ndarray
    dist_per_predator: jnp.ndarray
    agent_in_bush: jnp.ndarray   # NEW — bool, [num_envs] when vmapped
```

In `collect_trajectories` where `StepInfo(...)` is constructed (developer locates via `grep -n "StepInfo(" src/models/recurrent_ppo_trainer.py`), add one wiring line:

```python
step_info = StepInfo(
    ...
    dist_per_neutral=info['dist_per_neutral'],
    dist_per_predator=info['dist_per_predator'],
    agent_in_bush=info['agent_in_bush'],
)
```

**Risk surface**: a single field add — minimal. Make sure the order matches the dataclass declaration order; mis-positional construction at the call site would type-check but silently swap fields. The developer must use **keyword args** at the construction site (the existing call already does for the per-tag fields — confirm and follow the same style).

### 2.5 `src/models/dreamer_v3_trainer.py`

**Function**: transition dict (`src/models/dreamer_v3_trainer.py` around lines 609–633 — same site the per-tag plan modified). Add one entry:

```python
transition = {
    ...
    'dist_per_neutral':  info['dist_per_neutral'],
    'dist_per_predator': info['dist_per_predator'],
    'agent_in_bush':     info['agent_in_bush'],          # NEW — bool
    'termination_reason': info['termination_reason'].astype(jnp.float32),
}
```

Note: `agent_in_bush` is a `bool` JAX array; the existing `'event_collided'` and `'ate_food'` entries follow the same bool pattern with no cast. Keep `agent_in_bush` as bool (do not cast to float32) — the downstream accumulator treats it as a 0/1 integer.

**Risk surface**: same as 2.4 — additive entry, no order dependency at the dict.

### 2.6 `train.py` — online M1, M2, M5 accumulators (the load-bearing site)

This is the largest change. Five wiring sites match the per-tag precedent. **Implementation decision (K-buffer)**: per-env numpy buffers at the host level, NOT JAX-vectorised inside the rollout. Rationale captured in §6 below; in short, host-side numpy aligns naturally with the existing 5-site numpy accumulator pattern AND with the Site-2 (DreamerV3 batch) leftover-after-done logic that the per-tag plan's verification report (deviation #2) called out.

#### 2.6.1 State to add at the accumulator-init site (single site, near line 949)

After the existing `episode_dist_per_*_sums` init at `train.py:953-954`, add:

```python
# === Behavior-measure toolkit v1: online M1/M2/M5 state ===
# Loaded once from config; None when behavior_measures: block is absent (backwards-compat).
from src.environment.config_loader import load_behavior_measure_cfg
bm_cfg = load_behavior_measure_cfg(config)                  # may be None
bm_enabled = bm_cfg is not None and bm_cfg.enabled
if bm_enabled:
    bm_R = float(bm_cfg.cue_radius)
    bm_K = int(bm_cfg.obs_window)
else:
    bm_R = 0.0
    bm_K = 0

# Per-env, per-class M1/M2/M5 counters (all numpy int64; per-class fan-out is "predator" / "rabbit").
# Episode-life arrays — reset at done; stage-wipe handled at the stage transition.
_NUM_CLASSES = 2  # 0 = predator, 1 = rabbit
m1_candidates  = np.zeros((num_envs, _NUM_CLASSES), dtype=np.int64)
m1_interrupted = np.zeros((num_envs, _NUM_CLASSES), dtype=np.int64)
m2_onsets      = np.zeros((num_envs, _NUM_CLASSES), dtype=np.int64)
m2_dives       = np.zeros((num_envs, _NUM_CLASSES), dtype=np.int64)
m5_threat_steps = np.zeros((num_envs, _NUM_CLASSES), dtype=np.int64)
m5_safe_steps   = np.zeros((num_envs, _NUM_CLASSES), dtype=np.int64)
m5_eat_threat   = np.zeros((num_envs, _NUM_CLASSES), dtype=np.int64)
m5_eat_safe     = np.zeros((num_envs, _NUM_CLASSES), dtype=np.int64)

# Per-tag accumulators (mirror the per-class ones; lengths = num_*_for_log).
m1_candidates_tag  = np.zeros((num_envs, num_predator_for_log + num_neutral_for_log), dtype=np.int64)
m1_interrupted_tag = np.zeros((num_envs, num_predator_for_log + num_neutral_for_log), dtype=np.int64)
m2_onsets_tag      = np.zeros((num_envs, num_predator_for_log + num_neutral_for_log), dtype=np.int64)
m2_dives_tag       = np.zeros((num_envs, num_predator_for_log + num_neutral_for_log), dtype=np.int64)
m5_threat_steps_tag = np.zeros((num_envs, num_predator_for_log + num_neutral_for_log), dtype=np.int64)
m5_safe_steps_tag   = np.zeros((num_envs, num_predator_for_log + num_neutral_for_log), dtype=np.int64)
m5_eat_threat_tag   = np.zeros((num_envs, num_predator_for_log + num_neutral_for_log), dtype=np.int64)
m5_eat_safe_tag     = np.zeros((num_envs, num_predator_for_log + num_neutral_for_log), dtype=np.int64)
# Per-tag layout: first num_predator_for_log entries = predator tags (by index), then neutral tags.
_pred_slice    = slice(0, num_predator_for_log)
_neutral_slice = slice(num_predator_for_log, num_predator_for_log + num_neutral_for_log)

# K-buffer state for M1 (look-ahead: did the agent stop eating in the next K steps?)
# We carry an integer ring buffer per (env, class) storing the step indices where
# a candidate fired; at the next K steps when 'not ate_food' is observed, the
# oldest candidate within K is resolved as interrupted.
#
# Simpler equivalent implementation that we use here: a per-env deque-of-candidate-steps
# (Python list, capped length K, one per class) plus a running counter of "steps since
# the last ate_food event". When step t+K arrives, every candidate whose age == K
# is resolved (interrupted iff the agent has not eaten between t and t+K).
#
# For M2 the look-ahead is dual: a per-env deque-of-onset-steps; at step t+K each
# unresolved onset checks whether agent_in_bush has been True any time in (t, t+K].
m1_candidate_age = np.full((num_envs, _NUM_CLASSES), -1, dtype=np.int32)        # -1 = no pending candidate
m1_candidate_tag = np.full((num_envs, _NUM_CLASSES), -1, dtype=np.int32)        # tag-index of pending candidate (per-tag fan-out)
m1_steps_since_eat = np.zeros((num_envs,), dtype=np.int32)                      # incremented every step; reset on ate_food
m2_onset_age = np.full((num_envs, _NUM_CLASSES), -1, dtype=np.int32)            # -1 = no pending onset
m2_onset_tag = np.full((num_envs, _NUM_CLASSES), -1, dtype=np.int32)
m2_in_bush_seen = np.zeros((num_envs, _NUM_CLASSES), dtype=bool)                # has agent entered bush since onset?
# Prior-step "threat in radius" state (per env, per class) — needed for M2 onset detection.
m_prev_threat_in_R = np.zeros((num_envs, _NUM_CLASSES), dtype=bool)
m_prev_threat_in_R_tag = np.zeros((num_envs, num_predator_for_log + num_neutral_for_log), dtype=bool)
```

**Note on "one pending candidate at a time" simplification**: the spec in design doc §1.1 M1/M2 allows multiple overlapping candidates within K. The simpler implementation above uses a single slot per (env, class) — when a new candidate fires while one is still pending, we resolve the old one (count as not-interrupted unless eating already stopped) and overwrite. This is a documented approximation; if a future analysis shows it materially undercounts, upgrade to a per-(env, class) deque of size K. The same simplification applies to M2. For v1 this is acceptable because:
- The candidate density per episode is low (R = 3 cells; threats only briefly enter the cue radius);
- K = 5 is short, so overlap is rare in practice;
- The simplification is conservative (it lower-bounds the interruption/dive count, not the candidate count).

A flag controls whether to use the simple-slot or full-deque variant — locked to simple-slot in v1, documented in `behavior_measures.motif_window_K` adjacent comment (no new YAML key).

#### 2.6.2 Per-step accumulator block (5 sites, same pattern as per-tag distance)

The K-buffer / M1 / M2 / M5 updates run **per step, per env**. Each of the 5 sites already has its per-step section (look for `info_np['dist_per_neutral'][t]` pattern); insert a single function call:

```python
# After the existing per-tag distance accumulation block at each site,
# behind a single `if bm_enabled:` guard.

if bm_enabled and 'agent_in_bush' in info_np:
    _bm_step_update(
        info_np, t,           # info_np carries [num_envs] arrays this step
        m1_candidates, m1_interrupted, m1_candidate_age, m1_candidate_tag,
        m1_candidates_tag, m1_interrupted_tag, m1_steps_since_eat,
        m2_onsets, m2_dives, m2_onset_age, m2_onset_tag, m2_in_bush_seen,
        m2_onsets_tag, m2_dives_tag,
        m5_threat_steps, m5_safe_steps, m5_eat_threat, m5_eat_safe,
        m5_threat_steps_tag, m5_safe_steps_tag, m5_eat_threat_tag, m5_eat_safe_tag,
        m_prev_threat_in_R, m_prev_threat_in_R_tag,
        bm_R, bm_K,
        neutral_tags, predator_tags, _pred_slice, _neutral_slice,
    )
```

`_bm_step_update` is a helper to be defined alongside `_append_per_tag_means` (near line 956). Its body, in pseudocode:

```python
def _bm_step_update(info_np, t, ...):
    """One-step update of M1/M2/M5 counters across all envs, both classes.

    For each env:
      - Compute `threat_in_R_pred = min(dist_per_predator[t]) < R` (False if num_pred = 0)
      - Same for rabbit.
      - Compute `threat_in_R_tag[j]` per tag (per-instance < R).
      - Compute `near_tag_pred = argmin over predator-instance distances if threat_in_R_pred else -1`
      - Same for rabbit.

      M5: increment threat_steps OR safe_steps per class per env; if ate_food[t] increment
          eat_threat or eat_safe correspondingly. Per-tag uses threat_in_R_tag.

      M1: per-class:
        - If `ate_food[t] AND threat_in_R_<class>` and no pending candidate → record candidate
          at age 0; capture nearest-instance tag for the per-tag bucket.
        - Increment candidate ages by 1 each step (after recording).
        - If a candidate's age reaches K (i.e., reached step t+K) AND `m1_steps_since_eat[env] >= 1`
          since the candidate fired → count as interrupted. Either way, retire the candidate slot.
        - Update `m1_steps_since_eat[env] = 0 if ate_food[t] else m1_steps_since_eat[env] + 1`.

      M2: per-class:
        - If `not prev_threat_in_R[class] AND threat_in_R[class] AND not agent_in_bush[t]`
          → record onset (age 0; nearest-instance tag).
        - Increment onset ages by 1 each step.
        - If `agent_in_bush[t]` becomes True for any (env, class) with a pending onset → mark
          m2_in_bush_seen[env, class] = True.
        - When age reaches K → tally a dive iff m2_in_bush_seen is True; retire and clear.

    Update prev_threat_in_R / prev_threat_in_R_tag at the end of the step.
    """
```

A skeleton-ready implementation should be ~80 lines of numpy; vectorise across the env axis (no Python loops over `num_envs`). The class axis is just 2; a per-class Python loop is fine.

**Risk surface — the load-bearing point**:
- The 5 sites have different patterns for **how `info` reaches the host**:
  - Site 1 (RPPO main, `train.py:1191`): `info_np` is populated by a fixed-key-list loop. **This is the exact pattern that broke per-tag distance on the first ship** (the fix at `train.py:1194-1195` adds an explicit `info_np['dist_per_neutral'] = ...` line after the loop). The plan **mandates** the analogous explicit line for `agent_in_bush` here. Direct-extraction is structurally safer; the developer SHOULD also extend the `BEHAVIOR_KEYS`-style enumeration to include `'agent_in_bush'` so the existing loop picks it up — see §2.6.5.
  - Site 2 (DreamerV3 batch, `train.py:1438-1604`): uses `transitions_np.get('dist_per_neutral')` direct extraction. Add `transitions_np.get('agent_in_bush')`.
  - Sites 3–4 (DQN/DRQN steps): read `info` directly (JAX dict), so `info['agent_in_bush']` lands automatically.
  - Site 5 (Dreamer step): mirrors Site 1's `info_np` enumeration. Add `info_np['agent_in_bush'] = np.array(step_info.agent_in_bush)` after the existing loop, mirroring the per-tag fix.
- The **leftover-after-done** logic at Site 2 (per the per-tag verification deviation #2) means accumulators must be summed in chunks separated by done events. The M1/M2 K-buffer makes this trickier than per-tag distance: a candidate that fired in episode E with K = 5 unresolved steps left must NOT carry over into episode E+1 in the same env. **Fix**: on every `done` event for env i, **discard pending candidates / onsets** (do not count them as interrupted / dived). Implementation: in the same per-env reset block where `episode_dist_per_*_sums[i, :] = 0.0` is written, also reset `m1_candidate_age[i, :] = -1`, `m1_candidate_tag[i, :] = -1`, `m2_onset_age[i, :] = -1`, `m2_onset_tag[i, :] = -1`, `m2_in_bush_seen[i, :] = False`, `m_prev_threat_in_R[i, :] = False`, `m_prev_threat_in_R_tag[i, :] = False`, `m1_steps_since_eat[i] = 0`.
- The **stage-transition wipe** at `train.py:1107-1112` MUST extend to all the new arrays. One block, eight lines.

#### 2.6.3 Per-episode finalisation (5 sites)

In each per-env episode-end block (same 5 sites as per-tag), after the existing `ep_data['mean_dist_predator_<tag>_raw'] = ...` lines, compute the episode scalars:

```python
# Per-class M1
for c, cname in enumerate(("predator", "rabbit")):
    denom = m1_candidates[i, c]
    if denom > 0:
        ep_data[f"interrupted_feeding_rate_{cname}_raw"] = float(m1_interrupted[i, c]) / float(denom)
    else:
        ep_data[f"interrupted_feeding_rate_{cname}_raw"] = float("nan")
    ep_data[f"interrupted_feeding_denom_{cname}_raw"] = int(denom)

    denom = m2_onsets[i, c]
    if denom > 0:
        ep_data[f"bush_dive_rate_{cname}_raw"] = float(m2_dives[i, c]) / float(denom)
    else:
        ep_data[f"bush_dive_rate_{cname}_raw"] = float("nan")
    ep_data[f"bush_dive_denom_{cname}_raw"] = int(denom)

    threat_steps = m5_threat_steps[i, c]
    safe_steps   = m5_safe_steps[i, c]
    eat_threat   = m5_eat_threat[i, c]
    eat_safe     = m5_eat_safe[i, c]
    p_eat_threat = (eat_threat / threat_steps) if threat_steps > 0 else float("nan")
    p_eat_safe   = (eat_safe / safe_steps) if safe_steps > 0 else float("nan")
    EPS = 1e-6
    if threat_steps > 0 and safe_steps > 0:
        ep_data[f"eat_under_threat_ratio_{cname}_raw"] = float(p_eat_threat) / max(float(p_eat_safe), EPS)
    else:
        ep_data[f"eat_under_threat_ratio_{cname}_raw"] = float("nan")
    ep_data[f"eat_under_threat_rate_{cname}_raw"] = float(p_eat_threat) if threat_steps > 0 else float("nan")
    ep_data[f"eat_safe_rate_{cname}_raw"]         = float(p_eat_safe)   if safe_steps   > 0 else float("nan")
    ep_data[f"eat_under_threat_safe_steps_{cname}_raw"] = int(safe_steps)

# Per-tag M1, M2, M5 — analogous, using *_tag arrays and the predator_tags + neutral_tags layout.
for j_in_slice, tag in enumerate(predator_tags):
    j = _pred_slice.start + j_in_slice
    # ... per-tag M1/M2/M5 lines, same NaN-skip pattern as per-class ...
for j_in_slice, tag in enumerate(neutral_tags):
    j = _neutral_slice.start + j_in_slice
    # ... per-tag M1/M2/M5 lines ...
```

#### 2.6.4 Per-env reset block (5 sites)

After the existing `episode_dist_per_*_sums[i, :] = 0.0` lines, add zeroing for all 16 new arrays (8 per-class + 8 per-tag) plus the 7 ring-buffer slots:

```python
m1_candidates[i, :] = 0; m1_interrupted[i, :] = 0
m2_onsets[i, :] = 0;    m2_dives[i, :] = 0
m5_threat_steps[i, :] = 0; m5_safe_steps[i, :] = 0
m5_eat_threat[i, :] = 0;   m5_eat_safe[i, :] = 0
m1_candidates_tag[i, :] = 0; m1_interrupted_tag[i, :] = 0
m2_onsets_tag[i, :] = 0;    m2_dives_tag[i, :] = 0
m5_threat_steps_tag[i, :] = 0; m5_safe_steps_tag[i, :] = 0
m5_eat_threat_tag[i, :] = 0;   m5_eat_safe_tag[i, :] = 0
m1_candidate_age[i, :] = -1; m1_candidate_tag[i, :] = -1
m2_onset_age[i, :] = -1;     m2_onset_tag[i, :] = -1
m2_in_bush_seen[i, :] = False
m_prev_threat_in_R[i, :] = False
m_prev_threat_in_R_tag[i, :] = False
m1_steps_since_eat[i] = 0
```

Recommended: hoist into a helper `_bm_reset_env(i, *arrays)` so the 5-site copy stays in sync.

#### 2.6.5 Stage-transition wipe (single site, `train.py:1107-1112`)

Add the analogous 16 array wipes after the existing two `episode_dist_per_*_sums[:, :] = 0.0` lines. The K-buffer / age / steps arrays also reset.

#### 2.6.6 WandB ep_log fan-out (5 sites)

After the existing `_append_per_tag_means(ep_log, iteration_episodes, ...)` calls at each site, append new per-class and per-tag fan-outs for each measure. To avoid 5-site copy drift, generalise `_append_per_tag_means` (or add a sibling `_append_per_measure_means`):

```python
def _append_per_measure_mean(ep_log, iteration_episodes, ep_key_raw, wandb_key):
    """Mean the same per-episode raw scalar (skipping NaN) across the iteration's episodes."""
    vals = [ep[ep_key_raw] for ep in iteration_episodes
            if ep_key_raw in ep and not (isinstance(ep[ep_key_raw], float) and np.isnan(ep[ep_key_raw]))]
    if vals:
        ep_log[wandb_key] = float(np.mean(vals))
```

Per-site, the call list (per-class + per-tag, all 4 measures' WandB keys per design doc §2):

```python
# Per-class (always)
for cname in ("predator", "rabbit"):
    _append_per_measure_mean(ep_log, iteration_episodes,
        f"interrupted_feeding_rate_{cname}_raw", f"Episode/InterruptedFeedingRate_{cname}")
    _append_per_measure_mean(ep_log, iteration_episodes,
        f"interrupted_feeding_denom_{cname}_raw", f"Episode/InterruptedFeedingDenominator_{cname}")
    _append_per_measure_mean(ep_log, iteration_episodes,
        f"bush_dive_rate_{cname}_raw", f"Episode/BushDiveRate_{cname}")
    _append_per_measure_mean(ep_log, iteration_episodes,
        f"bush_dive_denom_{cname}_raw", f"Episode/BushDiveDenominator_{cname}")
    _append_per_measure_mean(ep_log, iteration_episodes,
        f"eat_under_threat_ratio_{cname}_raw", f"Episode/EatUnderThreatRatio_{cname}")
    _append_per_measure_mean(ep_log, iteration_episodes,
        f"eat_under_threat_rate_{cname}_raw", f"Episode/EatUnderThreatRate_{cname}")
    _append_per_measure_mean(ep_log, iteration_episodes,
        f"eat_safe_rate_{cname}_raw", f"Episode/EatSafeRate_{cname}")
    _append_per_measure_mean(ep_log, iteration_episodes,
        f"eat_under_threat_safe_steps_{cname}_raw", f"Episode/EatUnderThreatSafeSteps_{cname}")

# Per-tag (when tags exist)
for tag in predator_tags:
    _append_per_measure_mean(ep_log, iteration_episodes,
        f"interrupted_feeding_rate_predator_{tag}_raw", f"Episode/InterruptedFeedingRate_predator_{tag}")
    # ... (one block per measure × per-tag, same pattern)
for tag in neutral_tags:
    _append_per_measure_mean(ep_log, iteration_episodes,
        f"interrupted_feeding_rate_rabbit_{tag}_raw", f"Episode/InterruptedFeedingRate_rabbit_{tag}")
    # ... etc.
```

The developer should factor each measure-block into a helper (`_append_m1_keys`, `_append_m2_keys`, `_append_m5_keys`) to keep the 5-site copy mechanical.

#### 2.6.7 Backwards-compat / enabled-false path

When `bm_enabled` is False, every per-step / per-episode / per-env / stage-wipe block is gated by `if bm_enabled:` and is a no-op. The numpy arrays are still allocated (cost: ~num_envs × 32 int64s ≈ a few KB), but zero CPU time per step. **Verify with a smoke training** on an existing config that has no `behavior_measures:` block — `Episode/MeanDistRabbit_*` keys appear unchanged; no new keys emitted; no regression.

#### 2.6.8 `info_np` enumeration extension (single line)

At Sites 1 and 5 (the fixed-key-list pattern), extend the loop's list:

```python
# Site 1 (train.py:1191):
for k in BEHAVIOR_KEYS + BEHAVIOR_DIST_KEYS + ['termination_reason', 'agent_in_bush']:
    info_np[k] = np.array(getattr(step_info, k))
```

Equivalently, add an explicit `if bm_enabled: info_np['agent_in_bush'] = np.array(step_info.agent_in_bush)` line **after** the loop (matches the per-tag fix). Either is fine — the explicit line is what the synthetic-test-trap insight recommends. The verification protocol checks both Site 1 and Site 5.

### 2.7 `scripts/eval_rollout.py` — NEW FILE

Loads a frozen checkpoint, runs N deterministic episodes against the locked `behavior_measures.eval_seeds`, dumps per-episode `(state, action, info)` to disk.

**Shape**:

```
scripts/eval_rollout.py
    --run-tag <run_tag>                # e.g. "nm8gn7y2"
    --checkpoint-step <N>              # e.g. "10000000" (or "final")
    --config <config-yaml-path>        # the training config; behavior_measures block read from it
    --output-root <dir>                # default = behavior_measures.eval_output_root
```

Outputs to `<output-root>/<run_tag>/<checkpoint_step>/`:
- `metadata.json` — config snapshot, training step, wall-clock, JAX version, git commit, agent type (RPPO / Dreamer), checkpoint path.
- `episodes/<ep_idx>.npz` — per-episode dump (`agent_pos: [T, 2]`, `action: [T]`, `ate_food: [T]`, `agent_in_bush: [T]`, `dist_per_predator: [T, num_pred]`, `dist_per_neutral: [T, num_neutral]`, `hit_predator: [T]`, `hit_neutral: [T]`, `drive_hunger: [T]`, `drive_injury: [T]`, `termination_reason: scalar`, `seed: scalar`, `length: scalar`). `.npz` is the project's typical choice for per-step ndarrays (compressed, atomic, JAX-friendly). Parquet is the choice for tabular indexes (next bullet), not for nested per-step arrays.
- `windows/threat_onsets.parquet` — one row per threat-onset event:
  - `episode_idx`, `t_star` (onset step), `triggering_class` (str: `"predator"` or `"rabbit"`), `triggering_tag` (str), `window_start_t = t_star - 2`, `window_end_t = t_star + K_motif` (clipped to episode length).
- `online_replay.json` — M1/M2/M5 keys re-computed offline by replaying the dumps, for sanity-cross-check against the WandB log. Each key as a single number averaged across the N episodes (same NaN-skip as online).

**Checkpoint loading**: follow the existing flow training already uses. For RPPO, that is `RecurrentPPOTrainer.load_checkpoint(path)`; for Dreamer, the matching `DreamerTrainer.load_state(...)`. The script accepts a `--run-tag` and locates the latest (or specified) checkpoint under `results/<run_tag>/checkpoints/` — the developer reads the training code's checkpoint-save naming convention and mirrors it.

**Noise schedule replay**: `eval_obs_noise: training` means re-instantiate the env with the **same** `perceptual_noise` block that was active in the training config at the loaded checkpoint's step. For v1, training noise is constant across training (not annealed), so this reduces to "use the training config's noise block verbatim". Document this assumption; if a future study uses annealed noise, this needs revisit.

**Determinism**: `eval_policy_mode: deterministic` → argmax over logits (RPPO) or actor-mean (Dreamer). PRNG seeding: `jax.random.PRNGKey(behavior_measures.eval_seeds[ep_idx])` per episode; the env's stochasticity (rabbit jitter, predator patrol jitter) flows from that key. Re-running the same script with the same seed list MUST produce bit-identical outputs (this is the §5.5 failure-mode catalogue's sanity check).

**Risk surface**: this is a new script; main risk is checkpoint-loading mechanics. The developer should write a 5-line smoke first (`python scripts/eval_rollout.py --run-tag nm8gn7y2 --checkpoint-step 100`) to confirm the checkpoint loads and one episode runs, then expand to N = 200.

### 2.8 `scripts/motif_cluster.py` — NEW FILE

Reads `<output-root>/<run_tag>/<checkpoint_step>/episodes/*.npz` + `windows/threat_onsets.parquet`, computes 10 features per window, k-means clusters into 6 motifs, writes the analysis artefacts.

**Shape**:

```
scripts/motif_cluster.py
    --eval-root <output-root>/<run_tag>/<checkpoint_step>   # path to a single eval-rollout result
    --config <config-yaml-path>                              # for behavior_measures.* (K_motif, kmeans_k, seed, features, standardise)
```

Outputs in the same `motifs/` subdir:
- `feature_vectors.parquet` — one row per threat-onset window; columns = the 10 feature names from design doc §2.4 + the per-window metadata (episode_idx, t_star, triggering_class, triggering_tag).
- `cluster_assignments.parquet` — one row per window, with `cluster_label` (post-hoc string from the v1 vocabulary).
- `cluster_centroids.npy` — `[k=6, num_features=10]` array of centroids in z-scored space.
- `motif_distribution.json` — `{"freeze": 0.12, "flight": 0.34, ...}` fraction per motif.
- `silhouette.json` — `{"silhouette_mean": float, "per_cluster": [floats]}`.
- `exemplars.json` — `{"<motif>": [list-of-5 window indices nearest the centroid]}`.

**Featurisation**: each row of `threat_onsets.parquet` indexes into the corresponding `.npz`; the script slices `[window_start_t : window_end_t + 1]` and computes the 10 features verbatim per design doc §2.4. Use the triggering instance's tag (from `threat_onsets.parquet`) to pick the right column of `dist_per_<class>` for `min_threat_distance` and `threat_distance_change_rate`.

**Standardisation + clustering**: `sklearn.preprocessing.StandardScaler` for z-score (pooled or per-agent per `motif_standardise`), then `sklearn.cluster.KMeans(n_clusters=motif_kmeans_k, random_state=motif_kmeans_seed, n_init=10)`. sklearn — not JAX — because the clustering is offline, small (~600 rows), and reviewers expect sklearn.

**Labelling**: v1 leaves cluster labels as `cluster_0` … `cluster_5` in `cluster_assignments.parquet`. Human-labelling step is documented in the **verification protocol** (§5): the senior-developer / experiment-analyzer reads 5 exemplars per cluster and assigns the English names (`freeze`, `flight`, `bush_dive`, `freeze_then_flight`, `ignore`, `approach`). A second small script (`scripts/motif_label_apply.py`, NOT in v1 scope) can rewrite the parquet later if needed.

**Risk surface**: degenerate clusters (all 600 windows fall into 1 cluster) is a real possibility on the Cell A1 case where the agent has very few threat onsets and they all look identical. Handle: if any cluster has < 1% of windows, emit a warning in `silhouette.json` but do not error.

### 2.9 `tests/environment/test_behavior_measures.py` — NEW FILE

T1–T8 specified in §4 below.

### 2.10 `.claude/agents/env-config-auditor.md` — single-line addition

Append to the auditor's checklist under §2 (Mandatory-Key Discipline) or as a new §1.5 (Behavior-measures consistency):

```markdown
### 1.5 Behavior-measures Bush Presence (when `behavior_measures.enabled: true`)

- When `behavior_measures.enabled: true` AND M2 (bush_dive_rate) is in scope,
  verify `environment.obstacles` contains at least one entry with `hides_agent: true`.
  Without bushes, `info['agent_in_bush']` is always False, M2 emits NaN, and the
  measure is structurally uninterpretable.
```

This is a one-line addition per the user's directive.

---

## 3. Implementation Plan — step-by-step ordering

Dependency-aware. Each step is a logical commit boundary.

1. **Schema loader** (`src/environment/config_loader.py` — `BehaviorMeasureCfg` dataclass + `load_behavior_measure_cfg`). Test T1.
2. **`info['agent_in_bush']` env hook** (`src/environment/core.py`). Test T2.
3. **`StepInfo` + Dreamer transition dict** plumbing (`src/models/recurrent_ppo_trainer.py`, `src/models/dreamer_v3_trainer.py`).
4. **M5 online** (cheapest; just episode-level threat/safe step counters and ate-under-threat counters). Add the per-class accumulators, the per-episode finalisation, and the WandB fan-out at all 5 sites. Smoke train (T8 first pass — M5 keys only). This is the lowest-risk site to verify the 5-site pattern is right before stacking M1 + M2 on top.
5. **M1 + M2 online** (need the K-buffer). Add candidate / onset / age / steps_since_eat / in_bush_seen / prev_threat arrays; the per-step update helper; the per-episode finalisation; the WandB fan-out. Run T3, T4 unit tests. Re-run T8 — all 12 per-class keys + per-tag keys appear.
6. **`scripts/eval_rollout.py`** — the offline data source for M7. Smoke-load a Round-2.5 checkpoint and run 5 episodes (`--eval-n-episodes 5` override).
7. **`scripts/motif_cluster.py`** — consumes the eval-rollout output. Smoke on the 5-episode dump from step 6 with `motif_kmeans_k=3` (more clusters than data is degenerate) to confirm the pipeline runs end-to-end.
8. **T1–T8 tests pass** in `pytest tests/`.
9. **`env-config-auditor.md` addendum** (single-line addition).
10. **Real-`train.py` smoke training** (T8) on a small config (e.g., `02-sameProp_R2_passivePredator.yaml` with `behavior_measures.enabled: true`, RPPO, 4 envs, 3 iterations, CPU) — confirm all M1/M2/M5 keys appear in WandB / stdout with finite non-zero values. **The developer MUST run actual `train.py`, NOT a synthetic Python script** — see §5 verification protocol.

Commits should be one per major step (e.g., commit 1 = schema loader; commit 2 = env hook; commit 3 = StepInfo plumbing; commit 4 = M5 online; commit 5 = M1+M2 online; commit 6 = eval_rollout; commit 7 = motif_cluster; commit 8 = tests; commit 9 = auditor addendum).

---

## 4. Test plan (T1–T8)

`tests/environment/test_behavior_measures.py`:

### T1 — Schema loader rejects missing mandatory keys

For each of the 14 leaf keys in `behavior_measures:`, write a fixture YAML that omits that one key and assert `load_behavior_measure_cfg(config)` raises `ValueError` whose message mentions the omitted key. Parameterise with `pytest.mark.parametrize`. Also assert that the wholly-absent top-level `behavior_measures:` block returns `None` (backwards compat).

```python
@pytest.mark.parametrize("missing_key", [
    "behavior_measures.enabled", "behavior_measures.cue_radius",
    "behavior_measures.obs_window", "behavior_measures.eval_n_episodes",
    "behavior_measures.eval_seeds", "behavior_measures.eval_policy_mode",
    "behavior_measures.eval_max_steps", "behavior_measures.eval_obs_noise",
    "behavior_measures.motif_window_K", "behavior_measures.motif_features",
    "behavior_measures.motif_kmeans_k", "behavior_measures.motif_kmeans_seed",
    "behavior_measures.motif_standardise", "behavior_measures.eval_output_root",
])
def test_schema_loader_rejects_missing_key(tmp_path, missing_key):
    ...  # build yaml minus `missing_key`, assert ValueError raised
```

Plus targeted validation tests: `eval_seeds` length mismatch raises; `eval_policy_mode` typo raises; `motif_features` containing an unknown feature name raises; `cue_radius <= 0` raises.

### T2 — `info['agent_in_bush']` fires on bush cells

```python
def test_agent_in_bush_fires_at_known_obstacle():
    """Place agent at a known hides_agent=True obstacle; assert info['agent_in_bush'] is True."""
    # Build minimal env params with one bush at (3, 3), agent forced to step onto (3, 3).
    # Call jax_step with action that moves agent to (3, 3); assert info['agent_in_bush'] == True.
    # Move agent off the bush; assert info['agent_in_bush'] == False.
    ...
```

Uses a fixture config (similar to T2 of per-tag plan) — minimal grid + one `hides_agent: true` obstacle. Verify both fire (True at bush cell) and quiet (False elsewhere).

### T3 — M1 fires correctly on a fixture trajectory

```python
def test_m1_accumulator_on_fixture():
    """Hand-build a 20-step fixture trajectory: agent eats on steps 3, 5 with a
    predator within R on step 3, no predator on step 5. After step 3 + K = 8 the
    agent has stopped eating → candidate at step 3 resolves as interrupted.
    Step 5 (no threat) is not a candidate. Assert:
      m1_candidates_pred = 1, m1_interrupted_pred = 1 → rate = 1.0
      m1_candidates_rabbit = 0 → rate = NaN
    """
    # Drive the per-step update helper directly with hand-crafted info_np dicts.
    # Verify counters match the design doc §1.1 M1 formal predicate.
```

### T4 — M2 fires correctly on a fixture trajectory

```python
def test_m2_accumulator_on_fixture():
    """Hand-build a 15-step trajectory: predator enters R at step 4 (was outside
    at step 3), agent not in bush at step 4, agent enters bush at step 6.
    After step 4 + K = 9 the dive is resolved as fired. Assert:
      m2_onsets_pred = 1, m2_dives_pred = 1 → rate = 1.0
    """
    ...
```

### T5 — M5 ratio bounded; equals 1.0 on uniform-threat fixture

```python
def test_m5_ratio_uniform():
    """Hand-build a 40-step trajectory: predator within R on exactly half the
    steps; agent eats on 20% of steps regardless of threat state. Assert
    eat_under_threat_ratio_predator ≈ 1.0 (within 1e-6).
    Also assert: predator never in R → ratio = NaN (denominator zero).
    """
    ...
```

### T6 — Motif clustering produces exactly k = 6 clusters on a non-degenerate input

```python
def test_motif_clustering_smoke(tmp_path):
    """Generate 200 synthetic windows with 6 known motif archetypes (Gaussian
    feature clusters), run scripts/motif_cluster.py logic on them, assert:
      - All 6 clusters have at least 5 windows each (non-degenerate)
      - silhouette_mean > 0.20 (R4 sanity)
      - motif_distribution.json sums to 1.0
    """
    ...
```

This is a unit test on the clustering logic, NOT a full integration test — the synthetic windows are constructed in-test, not loaded from disk.

### T7 — Backwards compat: existing keys unchanged when `behavior_measures` absent

```python
def test_backwards_compat_no_behavior_measures_block():
    """Load a config WITHOUT a behavior_measures: block (e.g., a Round-1 config),
    assert load_behavior_measure_cfg(config) returns None. Run a 1-iteration
    smoke RPPO training; assert the existing Episode/MeanDistRabbit_TL,
    Episode/MeanDistPredator_TL keys appear with finite values; assert NO new
    Episode/InterruptedFeedingRate_* keys are emitted.
    """
    ...
```

### T8 — Real-`train.py` smoke training (mandatory, NOT a synthetic script)

```python
def test_t8_real_train_py_smoke(tmp_path):
    """Subprocess call to actual train.py with a minimal config + behavior_measures
    block. Parse the stdout / WandB log for the new keys. Assert all 12 per-class
    keys present with finite values (NaN allowed where denominators are 0); assert
    per-tag keys for predator_TL, rabbit_TL, rabbit_BR present.
    """
    cmd = [
        "/home/vncuser/miniconda3/envs/grid_world_pain/bin/python",
        "train.py",
        "--env-config", "configs/experiment/behavior_measures/smoke_test.yaml",
        "--agent-config", "configs/models/recurrent_ppo.yaml",
        "--num-envs", "4",
        "--max-iterations", "3",
        "--no-wandb",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, f"train.py failed:\n{out.stderr}"
    # Parse stdout for emitted keys; assert presence of:
    #   Episode/InterruptedFeedingRate_predator, _rabbit, _predator_TL, _rabbit_TL, _rabbit_BR
    #   Episode/BushDiveRate_predator, _rabbit, _predator_TL, _rabbit_TL, _rabbit_BR
    #   Episode/EatUnderThreatRatio_predator, _rabbit, _predator_TL, _rabbit_TL, _rabbit_BR
    # Assert each value is finite OR NaN (not negative, not inf).
```

**The smoke config** (`configs/experiment/behavior_measures/smoke_test.yaml`) — the experiment-designer's second pass writes this; for the developer's smoke during implementation, pick **`02-sameProp_R2_passivePredator.yaml`** and add an inline `behavior_measures:` block via a tmp-path patch. The developer surfaces the patched config diff in the Implementation Report so the senior-developer can verify.

**This test is the load-bearing protection against the synthetic-test trap.** Per the §5 verification protocol below, the developer MUST run actual `train.py` for T8, not a standalone Python script that bypasses `info_np` assembly. The senior-developer re-runs T8 independently as part of verification.

---

## 5. Verification protocol

The senior-developer runs this after the developer reports `done`. Six checks; all must pass before sign-off.

### V1 — All 8 unit tests + the per-tag toolkit's 7 prior tests pass

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/ -x -q
# Expect: 15+/15+ tests pass.
```

### V2 — T8 smoke training run on **actual** `train.py` output, NOT a synthetic script

This is the explicit avoidance lesson from [`20260509_1534_synthetic_smoke_masks_dict_assembly_bugs`](../../../../.claude-memory/memories/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md). The senior-developer:

1. Reads the developer's Implementation Report and locates the literal `train.py` command they ran.
2. Independently re-runs the same command (or a near-equivalent with different `--num-envs`).
3. Parses the stdout `Episode/...` lines (or the offline WandB summary) for all new keys.
4. Asserts each value is in the expected range (M1, M2 in `[0, 1] ∪ {NaN}`; M5 in `[0, ∞) ∪ {NaN}`; M1/M2 denominators are non-negative integers).
5. Asserts the aggregated `Episode/MeanDistRabbit` and per-tag keys from the per-tag toolkit are **numerically unchanged** vs a pre-implementation run on the same seed (i.e., no regression to existing keys).

**Hard rule**: if the developer's T8 evidence comes from a standalone `.py` file rather than `train.py`, V2 fails outright and the implementation is sent back regardless of unit-test pass.

### V3 — All 5 wiring sites verified by code-read

The senior-developer reads each of the 5 sites in `train.py` and confirms:
- Site 1 (RPPO main, ~line 1191): the `info_np` enumeration includes `'agent_in_bush'` AND each new measure's per-step update is called. **Specifically check that the dict-assembly is direct-extraction (`info_np['agent_in_bush'] = ...` after the loop) and NOT a buried fixed-key-list addition — the lesson from the per-tag bug.**
- Site 2 (Dreamer batch, ~line 1438): `transitions_np.get('agent_in_bush')` direct extraction.
- Sites 3, 4 (DQN, DRQN): `info['agent_in_bush']` direct access from the JAX dict.
- Site 5 (Dreamer step, ~line 2055): explicit `info_np['agent_in_bush'] = ...` after the fixed-key loop, mirroring the per-tag fix.

For each site, also confirm:
- Per-env reset block extends to all 16+7 new arrays.
- Stage-transition wipe extends to all 16+7 new arrays.
- WandB fan-out emits the right keys (per-class + per-tag) for each measure.

### V4 — M7 cluster exemplar review

Read 5 random exemplar windows per cluster from `cluster_assignments.parquet` + the source `.npz` files. For each, label by hand using the v1 motif vocabulary (`freeze`, `flight`, `bush_dive`, `freeze_then_flight`, `ignore`, `approach`). Compare against the script's cluster_label assignments. **Acceptance**: at least 4 of 6 clusters match the expected motif on hand inspection (the design doc's predicted dominant motifs per agent in §9 are the priors). If 3+ clusters look noisy, file a follow-up issue but do not block the merge — V1 is the demo; revisit features in v2.

This is the human-labelling step that v1 leaves out of `motif_cluster.py`; it lives in the verification protocol so the labels in the paper figure are reviewed.

### V5 — Backwards-compat: per-tag toolkit re-runs cleanly on Round-2.5 Cell A1 / Cell C checkpoints

Run `scripts/eval_rollout.py` on the saved `nm8gn7y2` and `bdnfc0lu` checkpoints (without M1/M2/M5 — just the env step path). Confirm the per-tag distance keys (`MeanDistRabbit_TL`, `MeanDistPredator_TL`, etc.) re-produce the values logged at training time. This is the strongest backward-compat check: a regression in the env step path would silently change per-tag distance numerics, which the eval-rollout would expose.

### V6 — Speed check

100-iteration RPPO walltime on `01-interoNocicept_sameProp.yaml` (the project's standard speed-check config), same hardware/seed, before vs after the changes. **Expected ≤ 2% slowdown** (the M1/M2/M5 numpy ops are O(num_envs × num_classes) per step, trivial); **>5% is a discussion point** (likely worth investigating); **>15% blocks merge**. If the developer skipped the speed check, the senior-developer runs it before sign-off.

After V1–V6, the senior-developer fills the Verification Report table per the issue_plan template and logs a row to the daily diary via `scripts/diary_append.py verified`.

---

## 6. Open questions / risks

### 6.1 K-buffer implementation: per-env numpy at host level (DECISION)

**Decision: per-env numpy buffers at the host level, NOT JAX-vectorised inside the rollout.**

Rationale:
- The existing 5-site accumulator pattern is **already** host-side numpy (`episode_dist_per_*_sums = np.zeros(...)`). Putting the K-buffer in JAX would require adding new pytree state to `StepInfo` / the Dreamer transition dict and threading per-env buffers through `lax.scan` — more code, more bug surface, and harder to align with Site 2's leftover-after-done logic (deviation #2 of the per-tag verification report).
- The K-buffer is tiny (K = 5 per env per class per measure) — the cost of host-side bookkeeping is negligible compared to the JAX env step.
- Numpy lets us cleanly **discard pending candidates on done** (a single integer reset per env per class) without threading "did this env's episode end?" through the JAX scan.
- This decision means Site 2's per-step accumulation runs through the `transitions_np` chunks the per-tag plan already established — the same code path that verified clean for per-tag distance.

The JAX-vectorised alternative was rejected because:
- It would require adding `m1_candidate_age`, `m2_onset_age`, `m_prev_threat_in_R`, etc., as pytree state, which means `StepInfo` grows by ~7 fields per measure × 2 classes — significant complexity.
- Done-handling in JAX (clear buffers on `done`) needs `jnp.where`-style masking inside the scan; in numpy it's a single `arr[done_mask, :] = 0`.
- Speed-up benefit is essentially zero — the bottleneck on RPPO is the JAX policy step, not the host accumulator (the per-tag toolkit's < 1% measured overhead confirms this).

### 6.2 `eval_obs_noise: training` semantics

Locked to "use the training-config's noise block verbatim" for v1. **Open question for the user**: confirm this matches expectation, or override (e.g., `eval_obs_noise: zero` is the cleaner held-out test for motif characterisation). Flagged for confirmation before the v1 demonstration runs.

### 6.3 Disk-budget estimate

Per checkpoint:
- 200 episodes × ~500 steps × ~50 floats per step × ~50 bytes per float (npz-compressed) ≈ **125–250 MB**.
- `threat_onsets.parquet`: ~600 rows × ~10 columns × ~50 bytes ≈ < 1 MB.
- `motifs/`: feature_vectors.parquet < 1 MB; cluster_centroids.npy ~few KB; JSON < 10 KB.
- **Total per checkpoint: ~250 MB.** For the v1 demo on 3 checkpoints (Cell A1 final, Cell C final, Round-1 baseline final) → **~750 MB.** Within the existing `results/` budget.
- **Threat-window-only would cut this 30×** (per design doc §3.3) but loses the offline sanity-replay; the design doc accepts the 150 MB price for the replay flexibility. **The plan honours that decision** — full-episode `.npz` dumps + `threat_onsets.parquet` window index.

### 6.4 Candidate-deque vs single-slot (M1, M2 implementation simplification)

The per-(env, class) simple-slot K-buffer (§2.6.1) is a conservative lower-bound approximation. If a follow-up analysis shows it materially under-counts candidates (e.g., > 5% of true candidates get overwritten before resolution), upgrade to a length-K deque in v2. **Risk for v1**: low — R = 3 cells + K = 5 + threat-onset density means overlapping candidates are rare in practice. Documented for the experiment-analyzer to spot-check after the first eval-rollout.

### 6.5 `predator_tags` / `neutral_tags` per-tag fan-out — same-tag deduplication

If two predators share a tag (e.g., `count: 2, tag: "TL"`), the per-tag M1/M2/M5 layout (one column per tag-instance) double-counts. The existing per-tag distance code averages over same-tag instances; the M1/M2 logic should follow the same convention. Note in the developer's per-tag block: when assembling the `_tag` per-step update, aggregate matching instances before testing the predicate (e.g., `min(dist_per_predator[t, j])` over `j` with the same tag) — match the per-tag distance reduction.

---

## 7. Cross-links to dependencies

- **Design doc** (`docs/experiments/active/behavior_measures/behavior_measure_toolkit_v1_design.md`) — §6 schema, §7 implementation surface, §1.1 formal predicates, §2 WandB key lists, §3 eval-rollout protocol.
- **Idea memo** (`docs/project/ideas/20260510_behavior_measure_toolkit.md`) — measure definitions and biological grounding for M1, M2, M5, M7.
- **Per-tag metric plan + verification** (`docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md`) — architectural precedent (the 5-site `train.py` accumulator pattern, the stage-transition wipe pattern, the WandB fan-out helper).
- **Synthetic-test-trap insight** (`.claude-memory/memories/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md`) — the load-bearing verification protocol rule that requires real-`train.py` smoke output, not a standalone script.
- **Frontmatter contract** (`docs/develop/active/meta/FRONTMATTER_CONTRACT.md`) — `topic` enum restriction; this plan files under `behavior/` (not `behavior_measures/`) for that reason. Flagged in Open Question 6.6 below.

### 6.6 Topic / folder placement

The user's task prompt asked for `docs/develop/active/behavior_measures/behavior_measure_toolkit_v1_plan.md` with `topic: behavior_measures`. The frontmatter contract restricts `topic` to the existing enum (`{behavior, hypervigilance, …}`), and `regen_dev_index.py` would reject an unknown topic. Adding `behavior_measures` to `VALID_TOPICS` requires editing `scripts/regen_dev_index.py`, which is outside this agent's no-implementation scope.

**Resolution chosen**: file under `docs/develop/active/behavior/` with `topic: behavior` (the existing valid topic for behavioural-measure work — co-located with `BEHAVIOR_ANALYSIS.md`, `TRAINING_METRICS_ANALYSIS.md`, `WANDB_METRICS_REFERENCE.md`). The file name is preserved as `behavior_measure_toolkit_v1_plan.md` so the cross-link from the design doc is the same modulo the directory.

**Override path**: if the user wants a dedicated `behavior_measures` topic (the measure family is large enough — eight candidates total in the menu, four shipping in v1, with more queued), they can ask `developer` to extend `VALID_TOPICS` in `scripts/regen_dev_index.py` and `FRONTMATTER_CONTRACT.md` in a single small commit, then this doc + the design doc get `git mv`'d under the new folder. Trivial follow-up.

---

## Checkpoints

The developer should verify each during implementation:

- [x] **C1 — Schema loader**: every `behavior_measures.*` mandatory key raises `ValueError` when omitted; full block returns a populated `BehaviorMeasureCfg`. T1 passes (14/14 parametrized + 6 validation sub-tests).
- [x] **C2 — `info['agent_in_bush']`**: T2 passes. `jax_reset` → `obs_pos` from state, bush_mask from params; info key present and returns valid bool.
- [x] **C3 — `StepInfo` / Dreamer transition dict**: `agent_in_bush` field added as 18th entry to RPPO `StepInfo` NamedTuple; `'agent_in_bush': info['agent_in_bush']` added to DreamerV3 transition dict.
- [x] **C4 — M5 standalone**: M5 counters wired at all 5 sites. T5/T5b pass. smoke training (T8) exits 0.
- [x] **C5 — M1 + M2 standalone**: T3 + T4 unit tests pass. M1 age-first ordering bug identified and fixed in both train.py and test standalone.
- [x] **C6 — Per-tag fan-out**: `_predator_TL`, `_rabbit_TL`, `_rabbit_BR` keys wired at all 5 sites. T8 (real train.py smoke) exits 0 with no KeyError/AttributeError in stderr.
- [ ] **C7 — `eval_rollout.py` 5-episode smoke**: script created at `scripts/eval_rollout.py`. Full smoke against a saved checkpoint not run in this session (requires checkpoint path); script verified syntactically importable.
- [ ] **C8 — `motif_cluster.py` smoke**: script created at `scripts/motif_cluster.py`. T6 skipped (sklearn absent in env). Script verified syntactically importable.
- [x] **C9 — Real-`train.py` 3-episode smoke (T8)**: `train.py` subprocess with smoke_test.yaml, 4 envs, 3 episodes → returncode 0, no BM errors in stderr. T8 PASS.
- [x] **C10 — `env-config-auditor.md` addendum**: §1.5 "Behavior-measures Bush Presence" added under the auditor's checklist.
- [x] **C11 — Speed**: BM enabled 21.9s vs BM disabled 20.8s (16 envs, 20 episodes on CPU) = +5.3% overhead. Exceeds the "≤ 2% expected" guideline but is within the "< 15% blocks merge" threshold. All BM ops are Python-side and gated by `if bm_enabled:`. Flagged for senior-developer review.

---

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-05-11

### Summary of changes (file-by-file)

**`src/environment/config_loader.py`**
- Added `from dataclasses import dataclass` and `from typing import Tuple` imports.
- Added `BehaviorMeasureCfg` frozen dataclass (14 fields) at module level.
- Added module-level constants `_ALLOWED_POLICY_MODES`, `_ALLOWED_NOISE_MODES`, `_ALLOWED_STANDARDISE`, `_DEFAULT_FEATURE_NAMES`.
- Added `load_behavior_measure_cfg(config) -> BehaviorMeasureCfg | None` — returns `None` if block absent (backwards compat); raises `ValueError` for any of the 14 missing/invalid keys.

**`src/environment/core.py`**
- Added `info['agent_in_bush']` after the `dist_per_*` assignments in `jax_step`. Uses `new_agent_pos` (post-step, not `state.agent_pos` pre-step) with `params.obs_hides_agent` mask. Empty-obstacle guard via conditional.

**`src/models/recurrent_ppo_trainer.py`**
- Added `agent_in_bush: jnp.ndarray` as 18th field of `StepInfo` NamedTuple.
- Added `agent_in_bush=info['agent_in_bush']` to `StepInfo(...)` constructor in `collect_trajectories`.

**`src/models/dreamer_v3_trainer.py`**
- Added `'agent_in_bush': info['agent_in_bush']` to the transition dict (between `dist_per_predator` and `termination_reason`).

**`train.py`**
- Import: `load_behavior_measure_cfg` added to existing import line.
- BM accumulator init block (~60 lines): `bm_cfg` / `bm_enabled` / `bm_R` / `bm_K`; per-class arrays [num_envs, 2] for M1/M2/M5; per-tag arrays [num_envs, num_pred + num_neut]; K-buffer state; `_pred_slice` / `_neutral_slice`.
- Helper functions: `_bm_reset_env(i)`, `_bm_step_update(info_np_t, done_mask)`, `_bm_finalise_episode(i, ep_data)`, `_bm_finalise_tag(i, j, tag, class_name, ep_data)`, `_append_per_measure_mean(ep_log, iteration_episodes, ep_key_raw, wandb_key)`, `_bm_log_wandb(ep_log, iteration_episodes)`.
- All 5 sites wired: per-step update, per-episode finalisation, per-env reset, stage-transition wipe, WandB fan-out.
- Site 1 (RPPO main): `info_np['agent_in_bush'] = np.array(step_info.agent_in_bush)` explicit extraction after the fixed-key-list loop (anti-Site-1-pattern).
- Site 5 (PPO non-recurrent): `if hasattr(step_info, 'agent_in_bush'):` guard (PPO StepInfo lacks this field).

**`configs/experiment/behavior_measures/smoke_test.yaml`**  (new file)
- 10×10 grid, 1 active predator (tag: TL), 2 rabbits (TL, BR), 6 bushes (3 per region). Full `behavior_measures:` block with all 14 keys.

**`tests/environment/test_behavior_measures.py`**  (new file)
- 28 tests collected: 14 parametrised T1 (missing-key), T1b-T1g (validation), T2 (bush hook), T3 (M1), T4 (M2), T5/T5b (M5), T6 (motif cluster, skipped — no sklearn), T7 (backwards compat), T8 (real train.py subprocess).

**`scripts/eval_rollout.py`**  (new file)
- Offline evaluation rollout: loads RPPO checkpoint, runs N deterministic episodes, dumps `.npz` + `threat_onsets.parquet` + `online_replay.json` + `metadata.json`. DreamerV3 loading raises `NotImplementedError` (pending).

**`scripts/motif_cluster.py`**  (new file)
- Offline M7 clustering: reads episodes + onset index, featurises 10 features per window, KMeans, writes `feature_vectors.parquet` + `cluster_assignments.parquet` + `cluster_centroids.npy` + `motif_distribution.json` + `silhouette.json` + `exemplars.json`.

**`.claude/agents/env-config-auditor.md`**
- Added §1.5 "Behavior-measures Bush Presence" under The Audit Checklist.

### Test results

```
Command: /home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/ -v
Result:  34 passed, 1 skipped in 58.09s
  - 27 new BM tests (T1–T8) all pass; T6 skipped (sklearn absent)
  - 7 existing tests (per_entity_info, per_tag_distance) all pass — no regressions
```

T8 (real train.py subprocess):
```
Command: /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/behavior_measures/smoke_test.yaml \
  --agent_config configs/models/recurrent_ppo.yaml \
  --num-envs 4 --episodes 3 --no-wandb --device cpu --quiet
returncode: 0 (22.8s)
stderr: no KeyError, no AttributeError
```

### Speed check

Config: `configs/experiment/behavior_measures/smoke_test.yaml`, 16 envs, 20 episodes, CPU.
- BM **enabled**:  21.9s total (`/usr/bin/time`)
- BM **disabled**: 20.8s total
- Delta: **+1.1s = +5.3% overhead** (exceeds ≤2% expected guideline; within <15% merge threshold)

Note: the smoke config is 10×10 with active predator — slightly heavier than a standard training config. The K-buffer Python loops dominate the overhead (not JAX). All BM ops are gated by `if bm_enabled:` so there is zero overhead when disabled.

### Bug fixed during implementation

**M1 age-first ordering**: In the original implementation, the candidate age was set to 0 and immediately incremented to 1 in the same step (`record then age` order). This caused age to reach `bm_K` after only `bm_K - 1` no-eat steps, while `steps_since_eat` would only be `bm_K - 1 < bm_K`. Result: the interrupted condition `steps_since_eat >= bm_K` would never fire.

Fix: reordered the M1 update to **age first, then record new candidates**. Freshly-set candidates (age=0) are not aged on the recording step; they receive their first increment on the subsequent step. After this fix, `bm_K` no-eat steps are needed for both age and `steps_since_eat` to reach `bm_K`, and the interrupted condition fires correctly. The fix was applied to both `train.py` `_bm_step_update` and the test standalone `_run_bm_step`.

### Deviations from plan

1. **T8 argument names**: The plan's T8 spec used `--env-config`/`--agent-config`/`--max-iterations` which do not match the actual `train.py` CLI (`--config`/`--agent_config`/`--episodes`). Corrected in the test.

2. **T2 obs_pos access**: Plan's T2 accessed `params.obs_pos` but `obs_pos` is in `EnvState` not `EnvParams`. Fixed by calling `jax_reset` first and reading `state.obs_pos`.

3. **M1 off-by-one bug**: Not in the plan (plan's pseudocode was ambiguous on ordering). Fixed as documented above. The test fixture was also corrected to remove an extraneous mid-fixture eat event that masked the bug.

4. **C7/C8 checkpoint smoke**: `eval_rollout.py` and `motif_cluster.py` created and syntactically correct, but full offline smoke (C7/C8) requires a saved checkpoint path not available in this session. Scripts are ready for the senior-developer to run against a Round-2.5 checkpoint.

5. **Speed overhead +5.3%**: Slightly above the "≤2% expected" guideline in V6. The smoke config is heavier than the standard speed-check config (`01-interoNocicept_sameProp.yaml`) — the plan's V6 spec uses that config for 100 iterations. Recommend senior-developer runs V6 with the canonical config before sign-off.

6. **T6 skipped (no sklearn)**: `scikit-learn` is not installed in the `grid_world_pain` conda env. T6 gracefully skips. To enable: `pip install scikit-learn`.

Implemented by: developer

---

## Verification Report

> **Verified by**: [senior-developer]
> **Date**: [YYYY-MM-DD]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/environment/config_loader.py` | `BehaviorMeasureCfg` + loader | | |
| `src/environment/core.py` | `info['agent_in_bush']` | | |
| `src/models/recurrent_ppo_trainer.py` | `StepInfo.agent_in_bush` | | |
| `src/models/dreamer_v3_trainer.py` | transition `agent_in_bush` | | |
| `train.py` Sites 1–5 — per-step / per-episode / per-env reset / stage-wipe / WandB fan-out for M1/M2/M5 | | | |
| `scripts/eval_rollout.py` | new file | | |
| `scripts/motif_cluster.py` | new file | | |
| `tests/environment/test_behavior_measures.py` | T1–T8 | | |
| `.claude/agents/env-config-auditor.md` | §1.5 addendum | | |

**Conclusion**: [one-line summary]

V1–V6 results:
- V1 (unit tests): [PASS / FAIL]
- V2 (real-`train.py` smoke): [PASS / FAIL]  ← load-bearing per the synthetic-test-trap insight
- V3 (5 wiring sites code-read): [PASS / FAIL]
- V4 (motif exemplar review, 5 per cluster × 6 clusters): [PASS / FAIL]
- V5 (backwards-compat per-tag keys unchanged): [PASS / FAIL]
- V6 (speed ≤ 2% regression): [PASS / WARN / FAIL]
