---
title: "Per-quadrant occupancy + per-rabbit-instance distance logging (Round-2 escalation)"
topic: hypervigilance
status: active
created: 2026-05-08
last_updated: 2026-05-08
phase: 1
---

# Per-quadrant occupancy + per-rabbit-instance distance logging

> **Status**: PLANNED
> **Opened**: 2026-05-08
> **Related**:
> - [`docs/experiments/active/hypervigilance/sameprop_round2_design.md`](../../../experiments/active/hypervigilance/sameprop_round2_design.md) — §7 Metrics Requested (origin), §9.5 (failure-mode mapping that motivates them), §9.8 step 2 (escalation), §9.10 (escalated specs as load-bearing).
> - [`docs/develop/active/hypervigilance/per_entity_avoidance_logging.md`](per_entity_avoidance_logging.md) — direct precedent (commit `4b55fc6`); this plan extends the same 5 pipeline stages additively.
> - [`docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`](sameprop_discriminating_channels.md) — channels memo, why quadrant-vs-class is the live ambiguity.

---

## Context

Round-2 Cell A1 (`02-sameProp_R2_passivePredator.yaml`) at 0.39 M episodes produced
`Δ ≡ MeanDistPredator − MeanDistRabbit = +3.86` cells — a 10× exceedance of the
H₁(A1) confirmation margin — but the result is **structurally uninterpretable**.
With predator quadrant-locked to TL `[[1,1],[5,5]]` and rabbits at TL+BR, an
agent that camps the BR rabbit corner and never visits TL produces *identical*
WandB numbers to an agent that learned class-conditional avoidance. The
aggregate `Episode/MeanDistRabbit = 2.58` could be "agent visits BR rabbit at
distance ~1.5 and TL rabbit at distance ~3.5" (avoidance of co-located
rabbit) **or** "agent visits both rabbits at distance ~2.6" (true class
indifference). The two pictures support opposite verdicts on H₀(A1) vs H₁(A1).

The fix is two additive metrics:

1. Per-rabbit-instance L2 distance, surfaced quadrant-tagged
   (`Episode/MeanDistRabbit_TL`, `..._BR`, etc.).
2. Per-episode quadrant occupancy fractions (`Episode/QuadrantOccupancy_{TL,TR,BL,BR}`).

Without these, no amount of additional training resolves the Cell A1 verdict
(see Round-2 §9.10). They are **prerequisites for the Round-2 re-launch**, not
optional polish.

## Analysis

### Existing pipeline (read-only audit)

The pipeline was already mapped end-to-end by the prior plan
[`per_entity_avoidance_logging.md`](per_entity_avoidance_logging.md) §Analysis;
this plan reuses every site identified there. Verbatim summary of the 5 stages
(verified against current `v1.3` source on 2026-05-08):

1. **Env step (`src/environment/core.py:431-502`)** — builds the per-step `info`
   dict. `dist_to_neutral` (the *aggregated* nearest-rabbit distance) is at
   `core.py:497`. Per-instance norms are already computed inside that line:
   `jnp.linalg.norm(state.neutral_pos - new_agent_pos, axis=-1)` produces a
   `[num_neutral]` vector and is then immediately `jnp.min`-reduced. To get
   per-instance, do not reduce.
2. **RPPO `StepInfo` NamedTuple (`src/models/recurrent_ppo_trainer.py:7-22`)**
   — populated in `collect_trajectories` near the existing `dist_to_neutral`
   wiring (~line 191-208 after the prior plan's expansion).
3. **Dreamer transition dict (`src/models/dreamer_v3_trainer.py:609-633`)** —
   mirrors `info`. New entries land next to the existing `dist_to_neutral` /
   `dist_to_hiding_predator` keys.
4. **`train.py` accumulators (`train.py:939-946`)** —
   - `BEHAVIOR_KEYS` (sum-aggregated boolean per-step events).
   - `BEHAVIOR_DIST_KEYS` (mean-aggregated per-step distances).
   - Used at: `1084-1086, 1166, 1192-1195, 1208-1211, 1223-1226, 1421,
     1439-1442, 1451-1454, 1462-1465, 1473-1476, 1482-1485, 1647-1650,
     1663-1666, 1674-1677, 1808-1811, 1824-1827, 1835-1838, 1929, 1940-1943,
     1955-1958, 1966-1969`.
5. **`train.py` WandB aggregation (5 mirrored sites)** — append
   `Episode/MeanDistRabbit_*` and `Episode/QuadrantOccupancy_*` next to the
   existing `Episode/MeanDistRabbit` line at:
   - PPO main: `train.py:1255`
   - PPO branch B: `train.py:1515`
   - Dreamer branch A: `train.py:1725`
   - Dreamer branch B: `train.py:1893`
   - Dreamer branch C: `train.py:2003`

### Why quadrant-tagged names (not generic `_idx0`, `_idx1`)

The Round-2 designer/user preference is **quadrant-tagged** — quadrant identity
is the experimentally meaningful axis. The mapping from a rabbit's static
`neutral_spawn_area` to a quadrant tag {TL, TR, BL, BR} is a deterministic
Python-time computation in the config loader and produces a static
`neutral_quadrant_idx` array on `EnvParams`. This solves four problems
simultaneously:

- **Generic across configs.** Any future config with rabbits in TR or BL
  (Round 3 candidates) gets meaningful keys without further code changes.
- **Stable WandB schema.** A given config emits a fixed key set every run.
  Configs with no TL rabbit simply never log `Episode/MeanDistRabbit_TL`.
- **JIT-safe.** The mapping happens once in `config_loader.py` (Python `int`s
  baked into a `jnp.ndarray` field on `EnvParams`); the env step does not
  branch on quadrant identity, only on indices.
- **Forward-compatible with multi-count YAML entries.** If a future YAML uses
  `count: 2, spawn_area: [[1,1],[5,5]]` (both rabbits in TL, expanded to two
  instances at config-load), both instances get tag `TL` and the per-quadrant
  mean averages over them — semantically what the user wants.

### Why not 4 boolean masks for QuadrantOccupancy

The §7 hint suggested 4 boolean masks per step. A simpler equivalent is one
`int32` field `quadrant_idx ∈ {0,1,2,3}` derived from `new_agent_pos`. At
episode-aggregation time `train.py` reduces the per-step trace into 4 fractions
via `np.mean(quadrant_idx == k)` for `k ∈ 0..3`. The information is identical;
the env step adds 1 scalar to `info` instead of 4 booleans, and the JIT graph
gains 2 integer divisions (`r // (h//2)`, `c // (w//2)`) plus a fused
`2*row_half + col_half` index — well under 1 % of step cost. The 4 boolean
expansion happens off-GPU in NumPy.

### Edge cases — quadrant boundary on a 10×10 grid

`params.height = params.width = 10`; positions are 0-indexed integer grid cells
(post-config-loader subtraction at `config_loader.py:70,118,168,209-210`).
Quadrants are then:

| Quadrant | Row range (incl) | Col range (incl) |
|---|---|---|
| TL (0) | 0–4 | 0–4 |
| TR (1) | 0–4 | 5–9 |
| BL (2) | 5–9 | 0–4 |
| BR (3) | 5–9 | 5–9 |

Splitter: `row_half = (new_agent_pos[0] >= height // 2).astype(int32)`,
`col_half = (new_agent_pos[1] >= width // 2).astype(int32)`,
`quadrant_idx = 2 * row_half + col_half`. For odd grid sizes (e.g. 9×9), the
center cell goes into the BR quadrant by `>=` — documented in a code comment.
The hypervigilance configs are all 10×10; this is a non-issue at the
experimental level.

For the per-rabbit quadrant tag (computed at config-load time in
`config_loader.py`), use the **midpoint of each `neutral_spawn_area`**:
`mid_r = (min_r + max_r - 1) / 2`, `mid_c = (min_c + max_c - 1) / 2` (the `-1`
because the loader stores `min_r-1, min_c-1, max_r, max_c` already; midpoint of
the *zero-indexed inclusive* range is `(min_r0 + max_r0 - 1) / 2` where
`min_r0 = min_r-1` is what's stored). Then apply the same `>= height//2`
splitter. This gives every YAML rabbit entry a single static quadrant tag,
even if its spawn area technically crosses the halfway line (which none of the
hypervigilance configs do — all rabbit areas are wholly within one quadrant).

### Backward compatibility

- No new YAML config keys → existing configs continue to load. (`config.get_mandatory`
  is not invoked.)
- `EnvParams` gains 1 new field (`neutral_quadrant_idx`, a `jnp.ndarray` of
  shape `[num_neutral]`, dtype int32). Existing `_replace`/dataclass field
  ordering is preserved (append at the bottom of the Neutral Animals block).
- `info` gains 2 new entries:
  - `dist_per_neutral` — `jnp.ndarray` of shape `[num_neutral]`, fixed at trace
    time. For configs with `num_neutral = 0` we fall back to a `(0,)`-shape
    array (handled identically to today's `state.neutral_pos.shape[0] > 0`
    guard).
  - `quadrant_idx` — `int32` scalar.
- `Episode/MeanDistRabbit` (the aggregated nearest-rabbit metric) is **not**
  removed; the new `Episode/MeanDistRabbit_TL` / `_BR` are additive.
- Per-quadrant rabbit keys appear only for quadrants that contain at least one
  rabbit. Configs with 0 rabbits emit none of the new rabbit keys; the
  `QuadrantOccupancy_*` keys always emit (4 keys, regardless of config) since
  the agent is always in some quadrant.
- The Dreamer replay buffer rebuilds itself from the transition dict at run
  start, so the new fixed-shape `dist_per_neutral` array propagates without
  buffer-shape surgery — same as the prior plan.

### Vmap / JIT safety

- `dist_per_neutral` operates on already-batched `state.neutral_pos` arrays
  and produces a fixed-shape `[num_neutral]` jax array — vmaps cleanly to
  `[num_envs, num_neutral]` like the existing `dist_to_neutral` reduction.
- `quadrant_idx` is a single int32 scalar derived from `new_agent_pos` and the
  static `params.height` / `params.width` — vmaps to `[num_envs]`.
- `neutral_quadrant_idx` lives on `EnvParams` as a regular `jnp.ndarray` (the
  dataclass currently mixes static `pytree_node=False` ints/floats with
  dynamic `jnp.ndarray` fields — see `state.py:65-115`); a `[num_neutral]`
  int array is shaped consistently with the existing `neutral_nociception` /
  `neutral_move_int` arrays, which are dynamic.
- All Python-level `if shape[0] > 0` guards stay at trace time (per the prior
  plan's analysis); recompile behaviour is unchanged.

## Implementation Plan

### Design

#### Final WandB metric set

| Key | Aggregation | Source `info` field | Notes |
|-----|-------------|---------------------|-------|
| `Episode/MeanDistRabbit_TL` | per-step mean of distances to rabbits whose static quadrant tag = TL | `dist_per_neutral` + `params.neutral_quadrant_idx` | Emitted only if ≥ 1 rabbit has tag TL. |
| `Episode/MeanDistRabbit_TR` | same, TR | same | Emitted only if ≥ 1 rabbit has tag TR. |
| `Episode/MeanDistRabbit_BL` | same, BL | same | Emitted only if ≥ 1 rabbit has tag BL. |
| `Episode/MeanDistRabbit_BR` | same, BR | same | Emitted only if ≥ 1 rabbit has tag BR. |
| `Episode/QuadrantOccupancy_TL` | fraction of episode steps with `quadrant_idx == 0` | `quadrant_idx` | Always emitted. Sums across 4 quadrants ≈ 1.0 (= 1.0 exactly for any individual env). |
| `Episode/QuadrantOccupancy_TR` | fraction, idx 1 | same | Always. |
| `Episode/QuadrantOccupancy_BL` | fraction, idx 2 | same | Always. |
| `Episode/QuadrantOccupancy_BR` | fraction, idx 3 | same | Always. |

For Round-2 A1/C the rabbit-quadrant set is `{TL, BR}` — the new keys are
`Episode/MeanDistRabbit_TL` and `Episode/MeanDistRabbit_BR` only. Future Round
3 configs with TR/BL rabbits get `_TR` / `_BL` automatically.

#### Per-step `info` additions

```python
# Two new info entries (to be inserted next to the existing `dist_to_neutral` block at core.py:497-501).
info['dist_per_neutral'] = (
    jnp.linalg.norm(state.neutral_pos - new_agent_pos, axis=-1)
    if state.neutral_pos.shape[0] > 0
    else jnp.zeros((0,), dtype=jnp.float32)
)
info['quadrant_idx'] = (
    2 * (new_agent_pos[0] >= (params.height // 2)).astype(jnp.int32)
    +     (new_agent_pos[1] >= (params.width  // 2)).astype(jnp.int32)
)
```

`dist_per_neutral.shape == (num_neutral,)`, which is a static, config-baked
size. `quadrant_idx` is a scalar `int32` in `{0,1,2,3}`. Neither field
triggers any new recompilation paths.

#### Static per-rabbit quadrant tag (config_loader)

In `src/environment/config_loader.py`, after `neutral_spawn_area` is built
(currently line 210), compute:

```python
# Quadrant tag per neutral (rabbit), based on spawn-area midpoint.
# Layout: 0=TL, 1=TR, 2=BL, 3=BR (matches `quadrant_idx` in core.py).
# Uses height/width loaded just above (lines 206-207).
if expanded_neutral:
    h_half = h // 2
    w_half = w // 2
    # neutral_spawn_area stores [min_r0, min_c0, max_r1, max_c1]
    # (min - 1 already applied by lines 209-210). Midpoint of the
    # zero-indexed inclusive range is (min_r0 + max_r1 - 1) / 2.
    mid_r = (neutral_spawn_area[:, 0] + neutral_spawn_area[:, 2] - 1) / 2
    mid_c = (neutral_spawn_area[:, 1] + neutral_spawn_area[:, 3] - 1) / 2
    row_half = (mid_r >= h_half).astype(jnp.int32)
    col_half = (mid_c >= w_half).astype(jnp.int32)
    neutral_quadrant_idx = 2 * row_half + col_half
else:
    neutral_quadrant_idx = jnp.zeros((0,), dtype=jnp.int32)
```

Then add `neutral_quadrant_idx=neutral_quadrant_idx` to the `EnvParams(...)`
constructor (around line 349, next to `neutral_spawn_area=`).

In `src/environment/state.py`, add the field to the `# Neutral Animals`
block of `EnvParams` (around line 108):

```python
neutral_spawn_area: jnp.ndarray  # [num_neutral, 4]
neutral_quadrant_idx: jnp.ndarray  # [num_neutral] int32 (0=TL, 1=TR, 2=BL, 3=BR; midpoint of spawn_area)
```

The new field is a regular dynamic `jnp.ndarray` (no `pytree_node=False`),
matching `neutral_spawn_area` directly above.

#### Train-time aggregation (per-quadrant fan-out)

`BEHAVIOR_DIST_KEYS` does **not** absorb `dist_per_neutral` (it's a vector,
not a scalar — the per-key sum loop at `train.py:1194` would NumPy-error).
Instead, add a parallel accumulator `episode_dist_per_neutral_sums` that
holds `(num_envs, num_neutral)` per-step sums, and a `quadrant_step_counts`
that holds `(num_envs, 4)` per-step quadrant counters.

```python
# train.py near line 946, after the existing accumulator dicts:
num_neutral_for_log = int(np.array(params.neutral_quadrant_idx).shape[0])
neutral_quadrant_np  = np.array(params.neutral_quadrant_idx, dtype=np.int32)  # (num_neutral,)
QUADRANT_NAMES       = ('TL', 'TR', 'BL', 'BR')
# Accumulators:
episode_dist_per_neutral_sums = np.zeros((num_envs, num_neutral_for_log), dtype=np.float32)
episode_quadrant_step_counts  = np.zeros((num_envs, 4), dtype=np.float32)
```

In each per-step accumulation block (5 sites; same as `BEHAVIOR_DIST_KEYS`
sites listed above), append:

```python
# After the existing BEHAVIOR_DIST_KEYS loop (~train.py:1194-1195):
if 'dist_per_neutral' in info_np and num_neutral_for_log > 0:
    episode_dist_per_neutral_sums += info_np['dist_per_neutral'][t]   # (B, num_neutral)
if 'quadrant_idx' in info_np:
    qidx_t = info_np['quadrant_idx'][t]  # (B,) int32
    # one-hot accumulate
    for q in range(4):
        episode_quadrant_step_counts[:, q] += (qidx_t == q).astype(np.float32)
```

In each per-episode-finalization block (5 sites; same locations as the
existing `for k in BEHAVIOR_DIST_KEYS: ep_data[k] = ...` loops), append per-quadrant
group means and per-quadrant occupancy fractions to `ep_data`:

```python
# After the existing dist-mean fill (~train.py:1210-1211):
ep_l = max(ep_length, 1)
# Per-rabbit-instance group means by quadrant tag.
if num_neutral_for_log > 0:
    means_per_neutral = episode_dist_per_neutral_sums[i] / ep_l  # (num_neutral,)
    for q in range(4):
        mask_q = (neutral_quadrant_np == q)
        if mask_q.any():
            ep_data[f'mean_dist_rabbit_q{q}'] = float(np.mean(means_per_neutral[mask_q]))
# Quadrant occupancy fractions.
quad_frac = episode_quadrant_step_counts[i] / ep_l  # (4,)
for q in range(4):
    ep_data[f'quadrant_occupancy_q{q}'] = float(quad_frac[q])
```

In each per-env reset block (right next to the existing
`episode_dist_sums[k][i] = 0.0` lines), append:

```python
episode_dist_per_neutral_sums[i, :] = 0.0
episode_quadrant_step_counts[i, :]  = 0.0
```

In each WandB ep_log block (5 sites listed above), append:

```python
# Right after the existing "Episode/MeanDistRabbit" line (~train.py:1255):
for q, qname in enumerate(QUADRANT_NAMES):
    key_dist = f'mean_dist_rabbit_q{q}'
    if any(key_dist in ep for ep in iteration_episodes):
        ep_log[f"Episode/MeanDistRabbit_{qname}"] = np.mean(
            [ep[key_dist] for ep in iteration_episodes if key_dist in ep]
        )
    ep_log[f"Episode/QuadrantOccupancy_{qname}"] = np.mean(
        [ep[f'quadrant_occupancy_q{q}'] for ep in iteration_episodes]
    )
```

The `if any(key_dist in ep for ep in iteration_episodes)` guard is what
suppresses `Episode/MeanDistRabbit_TR` / `_BL` for the Round-2 configs that
have no rabbits in those quadrants. `QuadrantOccupancy_*` is always emitted.

The mid-stage-transition wipe of accumulators at `train.py:1082-1087` (and
the equivalent Dreamer/Dreamer-branch wipes referenced through the same
constants) must be extended to wipe the two new accumulator arrays as well —
omitting this is the single most likely silent-bug source for this change
(stale Stage-N partial counters bleeding into Stage-N+1).

### File Changes

#### `src/environment/state.py` (line 108, inside `EnvParams.# Neutral Animals` block)

```python
# BEFORE (state.py:107-108):
    neutral_patrol: jnp.ndarray      # [num_neutral, 4]
    neutral_spawn_area: jnp.ndarray  # [num_neutral, 4]

# AFTER (add 1 new line below `neutral_spawn_area`):
    neutral_patrol: jnp.ndarray      # [num_neutral, 4]
    neutral_spawn_area: jnp.ndarray  # [num_neutral, 4]
    neutral_quadrant_idx: jnp.ndarray  # [num_neutral] int32 (0=TL,1=TR,2=BL,3=BR; midpoint of spawn_area)
```

#### `src/environment/config_loader.py` (around line 210, inside the `if expanded_neutral:` block; and constructor at line 349)

```python
# BEFORE (lines 209-210, end of the `if expanded_neutral:` block):
        neutral_patrol = jnp.array([[a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]] for a in [n.get('patrol_area', [[1,1],[h,w]]) for n in expanded_neutral]])
        neutral_spawn_area = jnp.array([[a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]] for a in [n.get('spawn_area', [[1,1],[h,w]]) for n in expanded_neutral]])

# AFTER (append static quadrant-tag computation; uses `h` / `w` already loaded at lines 206-207):
        neutral_patrol = jnp.array([[a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]] for a in [n.get('patrol_area', [[1,1],[h,w]]) for n in expanded_neutral]])
        neutral_spawn_area = jnp.array([[a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]] for a in [n.get('spawn_area', [[1,1],[h,w]]) for n in expanded_neutral]])
        # Quadrant tag per neutral, derived from spawn-area midpoint.
        # Layout: 0=TL, 1=TR, 2=BL, 3=BR (matches core.py `quadrant_idx`).
        # neutral_spawn_area entries are [min_r0, min_c0, max_r1, max_c1]
        # (min-1 already applied above). Midpoint of the zero-indexed inclusive
        # range = (min0 + max1 - 1) / 2; for odd grid sizes the boundary cell
        # goes BR by `>=`.
        h_half = h // 2
        w_half = w // 2
        _mid_r = (neutral_spawn_area[:, 0] + neutral_spawn_area[:, 2] - 1) / 2
        _mid_c = (neutral_spawn_area[:, 1] + neutral_spawn_area[:, 3] - 1) / 2
        neutral_quadrant_idx = (
            2 * (_mid_r >= h_half).astype(jnp.int32)
            +     (_mid_c >= w_half).astype(jnp.int32)
        )
```

```python
# In the matching `else` branch (lines 212-217), add:
        neutral_spawn_area = jnp.zeros((0, 4), dtype=jnp.int32)
        neutral_quadrant_idx = jnp.zeros((0,), dtype=jnp.int32)
```

```python
# In the `EnvParams(...)` constructor at line 349, add the new field next to `neutral_spawn_area=neutral_spawn_area,`:
        neutral_spawn_area=neutral_spawn_area,
        neutral_quadrant_idx=neutral_quadrant_idx,
```

#### `src/environment/core.py` (insert between lines 501 and 502, inside the GPU-distance block)

```python
# BEFORE (core.py:497-502):
    dist_to_neutral = jnp.min(jnp.linalg.norm(state.neutral_pos - new_agent_pos, axis=-1)) if state.neutral_pos.shape[0] > 0 else 99.0
    dist_to_hiding_predator = jnp.min(jnp.where(jnp.logical_and(state.res_active, params.res_type == 1), jnp.linalg.norm(state.res_pos - new_agent_pos, axis=-1), 99.0)) if state.res_pos.shape[0] > 0 else 99.0
    info['dist_to_food'] = dist_to_food
    info['dist_to_pred'] = dist_to_pred
    info['dist_to_neutral'] = dist_to_neutral
    info['dist_to_hiding_predator'] = dist_to_hiding_predator

# AFTER (add per-instance vector and quadrant scalar; existing keys unchanged):
    dist_to_neutral = jnp.min(jnp.linalg.norm(state.neutral_pos - new_agent_pos, axis=-1)) if state.neutral_pos.shape[0] > 0 else 99.0
    dist_to_hiding_predator = jnp.min(jnp.where(jnp.logical_and(state.res_active, params.res_type == 1), jnp.linalg.norm(state.res_pos - new_agent_pos, axis=-1), 99.0)) if state.res_pos.shape[0] > 0 else 99.0
    # Per-instance rabbit distances: keep the [num_neutral] vector (no reduction).
    dist_per_neutral = (
        jnp.linalg.norm(state.neutral_pos - new_agent_pos, axis=-1)
        if state.neutral_pos.shape[0] > 0
        else jnp.zeros((0,), dtype=jnp.float32)
    )
    # Agent's current quadrant: 0=TL, 1=TR, 2=BL, 3=BR (matches config_loader's neutral_quadrant_idx).
    quadrant_idx = (
        2 * (new_agent_pos[0] >= (params.height // 2)).astype(jnp.int32)
        +     (new_agent_pos[1] >= (params.width  // 2)).astype(jnp.int32)
    )
    info['dist_to_food'] = dist_to_food
    info['dist_to_pred'] = dist_to_pred
    info['dist_to_neutral'] = dist_to_neutral
    info['dist_to_hiding_predator'] = dist_to_hiding_predator
    info['dist_per_neutral'] = dist_per_neutral
    info['quadrant_idx'] = quadrant_idx
```

#### `src/models/recurrent_ppo_trainer.py` (`StepInfo` lines 7-22; constructor lines 191-208 after the prior plan)

```python
# BEFORE — StepInfo NamedTuple (after the prior plan landed): 15 fields.
class StepInfo(NamedTuple):
    """Per-step environment info carried through scan for behavioral logging."""
    ate_food: jnp.ndarray
    hit_predator: jnp.ndarray
    hit_hiding_predator: jnp.ndarray
    hit_neutral: jnp.ndarray
    event_collided: jnp.ndarray
    rested: jnp.ndarray
    damage: jnp.ndarray
    damage_predator: jnp.ndarray
    damage_hiding_predator: jnp.ndarray
    damage_obstacle: jnp.ndarray
    dist_to_food: jnp.ndarray
    dist_to_pred: jnp.ndarray
    dist_to_neutral: jnp.ndarray
    dist_to_hiding_predator: jnp.ndarray
    termination_reason: jnp.ndarray

# AFTER — add 2 new fields at the end (preserve existing order):
class StepInfo(NamedTuple):
    """Per-step environment info carried through scan for behavioral logging."""
    ate_food: jnp.ndarray
    hit_predator: jnp.ndarray
    hit_hiding_predator: jnp.ndarray
    hit_neutral: jnp.ndarray
    event_collided: jnp.ndarray
    rested: jnp.ndarray
    damage: jnp.ndarray
    damage_predator: jnp.ndarray
    damage_hiding_predator: jnp.ndarray
    damage_obstacle: jnp.ndarray
    dist_to_food: jnp.ndarray
    dist_to_pred: jnp.ndarray
    dist_to_neutral: jnp.ndarray
    dist_to_hiding_predator: jnp.ndarray
    termination_reason: jnp.ndarray
    dist_per_neutral: jnp.ndarray   # [num_neutral] — per-rabbit distance, unreduced
    quadrant_idx: jnp.ndarray       # int32 scalar — agent's quadrant
```

```python
# BEFORE — StepInfo construction in collect_trajectories (post-prior-plan):
step_info = StepInfo(
    ate_food=info['ate_food'],
    hit_predator=info['hit_predator'],
    hit_hiding_predator=info['hit_hiding_predator'],
    hit_neutral=info['hit_neutral'],
    event_collided=info['event_collided'],
    rested=info['rested'],
    damage=info['damage'],
    damage_predator=info['damage_predator'],
    damage_hiding_predator=info['damage_hiding_predator'],
    damage_obstacle=info['damage_obstacle'],
    dist_to_food=info['dist_to_food'],
    dist_to_pred=info['dist_to_pred'],
    dist_to_neutral=info['dist_to_neutral'],
    dist_to_hiding_predator=info['dist_to_hiding_predator'],
    termination_reason=info['termination_reason'],
)

# AFTER — add 2 new lines at the end (matching the new NamedTuple field order):
step_info = StepInfo(
    ate_food=info['ate_food'],
    hit_predator=info['hit_predator'],
    hit_hiding_predator=info['hit_hiding_predator'],
    hit_neutral=info['hit_neutral'],
    event_collided=info['event_collided'],
    rested=info['rested'],
    damage=info['damage'],
    damage_predator=info['damage_predator'],
    damage_hiding_predator=info['damage_hiding_predator'],
    damage_obstacle=info['damage_obstacle'],
    dist_to_food=info['dist_to_food'],
    dist_to_pred=info['dist_to_pred'],
    dist_to_neutral=info['dist_to_neutral'],
    dist_to_hiding_predator=info['dist_to_hiding_predator'],
    termination_reason=info['termination_reason'],
    dist_per_neutral=info['dist_per_neutral'],
    quadrant_idx=info['quadrant_idx'],
)
```

#### `src/models/dreamer_v3_trainer.py` (transition dict, lines 609-633)

```python
# BEFORE — Dreamer transition dict (post-prior-plan):
transition = {
    ...
    'dist_to_neutral': info['dist_to_neutral'],
    'dist_to_hiding_predator': info['dist_to_hiding_predator'],
    'termination_reason': info['termination_reason'].astype(jnp.float32),
}

# AFTER — add 2 new entries before `termination_reason`:
transition = {
    ...
    'dist_to_neutral': info['dist_to_neutral'],
    'dist_to_hiding_predator': info['dist_to_hiding_predator'],
    'dist_per_neutral': info['dist_per_neutral'],
    'quadrant_idx': info['quadrant_idx'].astype(jnp.float32),  # cast to float32 for replay-buffer homogeneity
    'termination_reason': info['termination_reason'].astype(jnp.float32),
}
```

Note: `dist_per_neutral` stays as `float32` (already is); `quadrant_idx` is
cast `int32 → float32` for Dreamer replay-buffer dtype consistency. The
`train.py` Dreamer aggregation sites use `info_np[k][t]` indexing directly off
the post-step `info` dict, not off the replay buffer, so the cast is purely a
buffer-dtype concern — does not affect aggregation values.

#### `train.py` (3 edits inside the lifecycle, applied at all 5 mirrored sites where applicable)

**Edit 1 — accumulator init (single-site, near line 946)**

```python
# BEFORE (train.py:945-946):
episode_behavior = {k: np.zeros(num_envs, dtype=np.float32) for k in BEHAVIOR_KEYS}
episode_dist_sums = {k: np.zeros(num_envs, dtype=np.float32) for k in BEHAVIOR_DIST_KEYS}

# AFTER — add 4 new lines (note: `params` here is the EnvParams from load_env_params at start-of-training):
episode_behavior = {k: np.zeros(num_envs, dtype=np.float32) for k in BEHAVIOR_KEYS}
episode_dist_sums = {k: np.zeros(num_envs, dtype=np.float32) for k in BEHAVIOR_DIST_KEYS}
# Per-rabbit + per-quadrant accumulators (Round-2 metrics).
num_neutral_for_log = int(np.array(params.neutral_quadrant_idx).shape[0])
neutral_quadrant_np = np.array(params.neutral_quadrant_idx, dtype=np.int32)  # (num_neutral,)
QUADRANT_NAMES      = ('TL', 'TR', 'BL', 'BR')
episode_dist_per_neutral_sums = np.zeros((num_envs, num_neutral_for_log), dtype=np.float32)
episode_quadrant_step_counts  = np.zeros((num_envs, 4), dtype=np.float32)
```

**Edit 2 — per-step accumulation (inside the `for t in range(num_steps):` loop, **5 sites**)**

Sites (mirrored — apply identical edit at each):
- `train.py:1187-1226` — PPO main, the existing block updates `episode_behavior` then `episode_dist_sums`. Append the new accumulators **after** `episode_dist_sums`.
- `train.py:1431-1485` — PPO branch B.
- `train.py:1645-1677` — Dreamer branch A.
- `train.py:1806-1838` — Dreamer branch B.
- `train.py:1927-1969` — Dreamer branch C.

Per-site insertion (representative, PPO main):

```python
# BEFORE (train.py:1187-1226 abbreviated):
for t in range(num_steps):
    episode_returns += rew_np[t]
    episode_lengths += 1
    if info_np:
        for k in BEHAVIOR_KEYS:
            episode_behavior[k] += info_np[k][t]
        for k in BEHAVIOR_DIST_KEYS:
            episode_dist_sums[k] += info_np[k][t]
    dones_t = done_np[t].astype(bool)
    if np.any(dones_t):
        completed_indices = np.where(dones_t)[0]
        for i in completed_indices:
            ...
            ep_data = {'r': ep_reward, 'l': ep_length}
            if info_np:
                for k in BEHAVIOR_KEYS:
                    ep_data[k] = float(episode_behavior[k][i])
                for k in BEHAVIOR_DIST_KEYS:
                    ep_data[k] = float(episode_dist_sums[k][i] / max(ep_length, 1))
                ep_data['termination_reason'] = int(info_np['termination_reason'][t][i])
            ep_info_buffer.append(ep_data)
            iteration_episodes.append(ep_data)
            # Reset for next episode in this slot
            episode_returns[i] = 0.0
            episode_lengths[i] = 0
            if info_np:
                for k in BEHAVIOR_KEYS:
                    episode_behavior[k][i] = 0.0
                for k in BEHAVIOR_DIST_KEYS:
                    episode_dist_sums[k][i] = 0.0

# AFTER — three additions, each marked with comments. Apply per-site.
for t in range(num_steps):
    episode_returns += rew_np[t]
    episode_lengths += 1
    if info_np:
        for k in BEHAVIOR_KEYS:
            episode_behavior[k] += info_np[k][t]
        for k in BEHAVIOR_DIST_KEYS:
            episode_dist_sums[k] += info_np[k][t]
        # === ROUND-2 METRICS: per-rabbit + per-quadrant accumulation ===
        if 'dist_per_neutral' in info_np and num_neutral_for_log > 0:
            episode_dist_per_neutral_sums += info_np['dist_per_neutral'][t]  # (B, num_neutral)
        if 'quadrant_idx' in info_np:
            qidx_t = info_np['quadrant_idx'][t]  # (B,) int (or float after Dreamer cast)
            qidx_t = qidx_t.astype(np.int32)
            for q in range(4):
                episode_quadrant_step_counts[:, q] += (qidx_t == q).astype(np.float32)
        # === END ROUND-2 METRICS ===
    dones_t = done_np[t].astype(bool)
    if np.any(dones_t):
        completed_indices = np.where(dones_t)[0]
        for i in completed_indices:
            ...
            ep_data = {'r': ep_reward, 'l': ep_length}
            if info_np:
                for k in BEHAVIOR_KEYS:
                    ep_data[k] = float(episode_behavior[k][i])
                for k in BEHAVIOR_DIST_KEYS:
                    ep_data[k] = float(episode_dist_sums[k][i] / max(ep_length, 1))
                ep_data['termination_reason'] = int(info_np['termination_reason'][t][i])
                # === ROUND-2 METRICS: per-episode finalization ===
                ep_l_safe = max(ep_length, 1)
                if num_neutral_for_log > 0:
                    means_per_neutral = episode_dist_per_neutral_sums[i] / ep_l_safe  # (num_neutral,)
                    for q in range(4):
                        mask_q = (neutral_quadrant_np == q)
                        if mask_q.any():
                            ep_data[f'mean_dist_rabbit_q{q}'] = float(np.mean(means_per_neutral[mask_q]))
                quad_frac = episode_quadrant_step_counts[i] / ep_l_safe  # (4,)
                for q in range(4):
                    ep_data[f'quadrant_occupancy_q{q}'] = float(quad_frac[q])
                # === END ROUND-2 METRICS ===
            ep_info_buffer.append(ep_data)
            iteration_episodes.append(ep_data)
            # Reset for next episode in this slot
            episode_returns[i] = 0.0
            episode_lengths[i] = 0
            if info_np:
                for k in BEHAVIOR_KEYS:
                    episode_behavior[k][i] = 0.0
                for k in BEHAVIOR_DIST_KEYS:
                    episode_dist_sums[k][i] = 0.0
                # === ROUND-2 METRICS: per-env reset ===
                episode_dist_per_neutral_sums[i, :] = 0.0
                episode_quadrant_step_counts[i, :]  = 0.0
                # === END ROUND-2 METRICS ===
```

The Dreamer-branch sites do not all use `info_np[...][t]` indexing identically
— branch A reads `info_np_step` via `info_np_step = {k: np.array(info[k]) for k
in ...}` at line 1647, then builds `ep_data` *outside* the per-step loop. Confirm
the per-site shape of `dist_per_neutral` (`(B, num_neutral)` after `np.array`)
and `quadrant_idx` (`(B,)`) before applying the same logic. The implementing
agent must verify each site reads `info_np[k][t]` vs `info_np_step[k]` and adjust
the indexing accordingly; the **arithmetic** is identical.

**Edit 3 — WandB ep_log block (5 sites, post-existing-MeanDistRabbit line)**

Sites (the same 5 sites that already log `Episode/MeanDistRabbit`):
- `train.py:1255` — PPO main
- `train.py:1515` — PPO branch B
- `train.py:1725` — Dreamer A
- `train.py:1893` — Dreamer B
- `train.py:2003` — Dreamer C

Per-site insertion:

```python
# BEFORE (representative, PPO main, lines 1253-1258):
"Episode/MeanDistFood": np.mean([ep['dist_to_food'] for ep in iteration_episodes]),
"Episode/MeanDistPredator": np.mean([ep['dist_to_pred'] for ep in iteration_episodes]),
"Episode/MeanDistRabbit": np.mean([ep['dist_to_neutral'] for ep in iteration_episodes]),
"Episode/MeanDistHidingPredator": np.mean([ep['dist_to_hiding_predator'] for ep in iteration_episodes]),
"Episode/RabbitHits": np.mean([ep['hit_neutral'] for ep in iteration_episodes]),
"Episode/HidingPredatorHits": np.mean([ep['hit_hiding_predator'] for ep in iteration_episodes]),

# AFTER — append per-quadrant fan-out via a small loop *after* the dict literal close.
# Implementation note: the existing 5 sites use `ep_log = { ... }` literal syntax;
# the loop must run after the literal is built. Two acceptable patterns:
#
#   Pattern A (preferred — fewer diffs): build the literal, then mutate it.
#       ep_log = { ... existing keys ... }
#       for q, qname in enumerate(QUADRANT_NAMES):
#           rkey = f'mean_dist_rabbit_q{q}'
#           if any(rkey in ep for ep in iteration_episodes):
#               ep_log[f"Episode/MeanDistRabbit_{qname}"] = float(np.mean(
#                   [ep[rkey] for ep in iteration_episodes if rkey in ep]
#               ))
#           ep_log[f"Episode/QuadrantOccupancy_{qname}"] = float(np.mean(
#               [ep[f'quadrant_occupancy_q{q}'] for ep in iteration_episodes]
#           ))
#
#   Pattern B: dict-merge with `**{ f"Episode/...": ... for q in ... }` inside the literal.
#       (More compact but harder to grep; avoid.)
#
# Choose Pattern A. Apply at all 5 sites identically.
```

The implementing agent must place the post-mutation block **before** the
`wandb.log(ep_log, ...)` call at the end of each block; for branch C (Dreamer)
the literal is `ep_logs.update({...})` rather than `ep_log = { ... }` — adjust
the variable name accordingly. The same guard `if any(rkey in ep for ep in
iteration_episodes)` correctly suppresses TR / BL keys for the hypervigilance
configs.

**Edit 4 — stage-transition wipe (single-site, near `train.py:1082-1087`)**

```python
# BEFORE:
episode_returns[:] = 0.0
episode_lengths[:] = 0
for _bk in BEHAVIOR_KEYS:
    episode_behavior[_bk][:] = 0.0
for _bk in BEHAVIOR_DIST_KEYS:
    episode_dist_sums[_bk][:] = 0.0

# AFTER (add 2 lines after the existing wipes):
episode_returns[:] = 0.0
episode_lengths[:] = 0
for _bk in BEHAVIOR_KEYS:
    episode_behavior[_bk][:] = 0.0
for _bk in BEHAVIOR_DIST_KEYS:
    episode_dist_sums[_bk][:] = 0.0
# === ROUND-2 METRICS: per-rabbit / per-quadrant wipes ===
episode_dist_per_neutral_sums[:, :] = 0.0
episode_quadrant_step_counts[:, :]  = 0.0
# === END ROUND-2 METRICS ===
```

`grep -n "episode_returns\[:\] = 0.0" train.py` will surface any other wipe
sites; if more exist (e.g., Dreamer-side stage-transition), add the same 2
lines next to each. Per the prior plan's audit there is only the one
stage-transition wipe at line 1082.

### Test plan

#### T1 — Smoke test: new info keys present and well-shaped

`tests/environment/test_per_quadrant_and_per_rabbit_info.py`:

```python
"""Per-quadrant + per-rabbit-instance info keys are present and JIT-safe."""
import jax
import jax.numpy as jnp
import numpy as np
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step

CONFIG = "configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml"

def test_dist_per_neutral_and_quadrant_idx_present_and_finite():
    params = load_env_params(CONFIG)
    key = jax.random.PRNGKey(0)
    state, _ = jax_reset(params, key)
    _, _, _, info = jax_step(state, jnp.array(4, dtype=jnp.int32), params)

    # dist_per_neutral has shape (num_neutral,) and is finite
    assert 'dist_per_neutral' in info
    dpn = np.array(info['dist_per_neutral'])
    assert dpn.shape == (int(np.array(params.neutral_quadrant_idx).shape[0]),)
    assert np.all(np.isfinite(dpn))

    # quadrant_idx is a finite int in {0,1,2,3}
    assert 'quadrant_idx' in info
    qidx = int(np.array(info['quadrant_idx']))
    assert qidx in (0, 1, 2, 3)

def test_neutral_quadrant_idx_matches_round2_layout():
    """02-sameProp_R2_passivePredator.yaml: rabbit 0 in TL (idx 0), rabbit 1 in BR (idx 3)."""
    params = load_env_params(CONFIG)
    nqi = np.array(params.neutral_quadrant_idx)
    assert nqi.tolist() == [0, 3], f"expected [TL=0, BR=3], got {nqi.tolist()}"
```

#### T2 — Quadrant fractions sum to 1.0 over a deterministic 10-step rollout

```python
def test_quadrant_occupancy_fractions_sum_to_one_over_episode():
    """A 10-step deterministic rest-rollout: agent stays at start_pos every step.
    Fraction of steps in agent's quadrant = 1.0; other 3 quadrants = 0.0."""
    params = load_env_params(CONFIG)
    key = jax.random.PRNGKey(123)
    state, _ = jax_reset(params, key)

    # Force a known agent position via state replace.
    state = state._replace(agent_pos=jnp.array([2, 2], dtype=state.agent_pos.dtype))  # TL

    qidx_seen = []
    for _ in range(10):
        state, _, _, info = jax_step(state, jnp.array(4, dtype=jnp.int32), params)  # rest
        qidx_seen.append(int(np.array(info['quadrant_idx'])))

    # Resting at (2,2) keeps the agent in TL (height//2=5, width//2=5 → row<5 and col<5).
    fracs = np.array([np.mean(np.array(qidx_seen) == q) for q in range(4)])
    assert np.isclose(fracs.sum(), 1.0)
    assert np.isclose(fracs[0], 1.0)  # TL
    assert np.allclose(fracs[1:], 0.0)
```

#### T3 — Per-rabbit distance equals L2 distance to that specific rabbit

```python
def test_per_rabbit_distance_matches_known_geometry():
    """Place agent at (2,2) (TL). MeanDistRabbit_TL after 1 step ≈ L2[(2,2), TL-rabbit-pos]."""
    params = load_env_params(CONFIG)
    key = jax.random.PRNGKey(7)
    state, _ = jax_reset(params, key)
    # TL rabbit: state.neutral_pos[0]; BR rabbit: state.neutral_pos[1]
    state = state._replace(agent_pos=jnp.array([2, 2], dtype=state.agent_pos.dtype))
    pre_neutral = np.array(state.neutral_pos)  # snapshot before the rabbit moves

    _, _, _, info = jax_step(state, jnp.array(4, dtype=jnp.int32), params)

    dpn = np.array(info['dist_per_neutral'])
    # dist_per_neutral is computed with state.neutral_pos PRE-step (matches existing
    # dist_to_neutral semantics at core.py:497). Expected:
    expected = np.linalg.norm(pre_neutral - np.array([2, 2]), axis=-1)
    assert np.allclose(dpn, expected, atol=1e-5), f"got {dpn}, expected {expected}"
```

If `dist_to_neutral` is actually computed against `state.neutral_pos` *after*
the rabbit move within the same `jax_step` (this is a question for the
implementer to verify by reading `core.py` around the rabbit-movement block),
relax T3 to `np.allclose(dpn, expected_post, atol=1.5)` — rabbit moves at most
1 cell per step, so the per-instance distance differs by at most √2 ≈ 1.42
from the pre-step position. Note T3 is the math-correctness gate; if it fails,
the implementation is wrong.

#### T4 — Dropping a quadrant: keys correctly suppressed

```python
def test_no_TR_BL_rabbits_suppresses_those_quadrant_keys():
    """The R2 configs have rabbits only in TL and BR. The fan-out logic should
    expose mean_dist_rabbit_q0 and _q3 in ep_data, and not _q1 / _q2."""
    # Pure-Python check on the static neutral_quadrant_idx — no env step needed.
    params = load_env_params(CONFIG)
    nqi = np.array(params.neutral_quadrant_idx)
    has_q = {q: bool((nqi == q).any()) for q in range(4)}
    assert has_q == {0: True, 1: False, 2: False, 3: True}
```

#### T5 — Smoke training run (post-merge sanity)

Run a 3-iteration smoke RPPO training on `02-sameProp_R2_passivePredator.yaml`
and confirm `Episode/QuadrantOccupancy_TL`, `..._TR`, `..._BL`, `..._BR`,
`Episode/MeanDistRabbit_TL`, `Episode/MeanDistRabbit_BR` appear in
`wandb/run-*/files/output.log` (or stdout `Episode/...` echoes). T5 is the
end-to-end gate; T1–T4 are unit gates.

### Backwards-compat / cross-config sanity

- `01-interoNocicept_sameProp.yaml` (Round 1) — same TL+BR rabbit layout;
  emits the same 6 new keys as Round 2. Aggregated `Episode/MeanDistRabbit`
  numerically unchanged.
- `01-interoNocicept.yaml`, `01-interoNocicept_noise.yaml` — same TL+BR layout
  per the config-survey grep above; same key set.
- Configs with 0 rabbits (none currently in `configs/experiment/hypervigilance/`,
  but historic configs may exist) — `num_neutral_for_log = 0`,
  `episode_dist_per_neutral_sums.shape = (num_envs, 0)`, no
  `Episode/MeanDistRabbit_*` keys emitted; `Episode/QuadrantOccupancy_*` still
  emitted (4 keys, sum to ~1.0 per episode).
- Configs with rabbits in TR or BL (Round-3 candidates) — `_TR` / `_BL` keys
  appear automatically without code changes.

## Checkpoints

What the implementing agent should verify **during** implementation:

- [ ] **C1 — `state.py`**: `EnvParams` builds without error after adding
  `neutral_quadrant_idx`. `python -c "from src.environment.state import EnvParams"` exits 0.
- [ ] **C2 — `config_loader.py`**: `load_env_params("configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml")` returns a params object with
  `np.array(params.neutral_quadrant_idx).tolist() == [0, 3]`. Same call on
  `01-interoNocicept_sameProp.yaml` returns `[0, 3]`. Same on `02-sameProp_R2_decoupleFood.yaml` returns `[0, 3]`.
- [ ] **C3 — `core.py`**: T1 passes (info keys present, finite, correctly shaped).
- [ ] **C4 — `recurrent_ppo_trainer.py`**: `StepInfo` has 17 fields; a
  1-iteration RPPO smoke runs without `KeyError` and `step_info.dist_per_neutral.shape ==
  (num_steps, num_envs, num_neutral)`.
- [ ] **C5 — `dreamer_v3_trainer.py`**: `DreamerTrainer` imports without error;
  the transition dict includes `dist_per_neutral` and `quadrant_idx` (verified
  by `grep -c "'dist_per_neutral'" src/models/dreamer_v3_trainer.py` returning ≥ 1).
- [ ] **C6 — `train.py` accumulators**: `grep -n "episode_dist_per_neutral_sums" train.py`
  returns 1 init line + 5 per-step accumulation sites + 5 per-env reset sites
  + 1 stage-wipe site = **12 hits**. `grep -n "episode_quadrant_step_counts" train.py`
  returns the same count = **12 hits**.
- [ ] **C7 — `train.py` WandB**: `grep -c '"Episode/QuadrantOccupancy_TL"' train.py`
  returns **5**. Same for `_TR`, `_BL`, `_BR`. Same for `"Episode/MeanDistRabbit_TL"`
  (5) and `_BR` (5).
- [ ] **C8 — Tests**: T1, T2, T3 (or relaxed T3), T4 pass.
- [ ] **C9 — Smoke training (T5)**: A 3-iteration RPPO smoke on
  `02-sameProp_R2_passivePredator.yaml` echoes all new `Episode/...` keys with
  finite values and the four `QuadrantOccupancy_*` fields summing to ≈ 1.0
  (per-iteration mean across episodes; tolerance 0.05 for partial-episode
  episodes captured by the iteration window).

## Acceptance Criteria

The implementing agent declares "done" when **all** hold:

1. `info['dist_per_neutral']` is a `[num_neutral]` jax.Array after every
   `jax_step`; `info['quadrant_idx']` is an int32 scalar in `{0,1,2,3}`.
2. `EnvParams.neutral_quadrant_idx` is a `[num_neutral]` int32 jax.Array,
   correctly mapped from `neutral_spawn_area` midpoints. T4 passes for the
   3 hypervigilance configs (all return `[0, 3]`).
3. `StepInfo` has the 2 new fields; `collect_trajectories` populates all 17
   fields without `KeyError`.
4. The Dreamer transition dict has `dist_per_neutral` and `quadrant_idx`.
5. `train.py` accumulators initialise correctly for `num_neutral = 0` and
   `num_neutral > 0`. The stage-transition wipe extends to both new arrays.
6. **All 5** WandB aggregation sites emit `Episode/QuadrantOccupancy_{TL,TR,BL,BR}` (4 keys × 5 sites = 20 hits) and emit
   `Episode/MeanDistRabbit_{TL,BR}` for the hypervigilance R2 configs (2 keys
   × 5 sites = 10 hits). `_TR` and `_BL` are correctly **not** emitted for
   those configs.
7. The aggregated `Episode/MeanDistRabbit` (the existing key from the prior
   plan) is **numerically unchanged** vs a pre-change run on the same config
   + seed for the first iteration's mean. (Sanity: the new code only adds
   accumulators; it does not alter the existing reduction.)
8. T1, T2, T4 pass. T3 passes either tight (`atol=1e-5`) or relaxed
   (`atol=1.5`) per the rabbit-move semantics — implementer notes which.
9. **No new YAML config keys.** `config.get_mandatory(...)` is **not** called
   for any new key.
10. **Speed-check note** in the Implementation Report: 100-iteration RPPO
    walltime on `01-interoNocicept_sameProp.yaml` before vs after, same
    hardware/seed. Expected ≤ 1 % slowdown; **>5 % is a blocker**.

## Implementation Report

> **Implemented by**: [agent/person]
> **Date**: [date]

<!-- Filled by the developer after code changes are made.
     Describe what was done, any deviations from the plan, and why. -->

## Verification Report

> **Verified by**: [agent/person]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/environment/state.py` | +1 EnvParams field | | |
| `src/environment/config_loader.py` | +static-quadrant-tag block, +1 constructor arg | | |
| `src/environment/core.py` | +2 info keys (`dist_per_neutral`, `quadrant_idx`) | | |
| `src/models/recurrent_ppo_trainer.py` | +2 StepInfo fields, +2 wiring lines | | |
| `src/models/dreamer_v3_trainer.py` | +2 transition keys | | |
| `train.py` (accumulators init + 5 per-step sites + 5 reset sites + 1 stage-wipe) | per-rabbit + per-quadrant accumulators | | |
| `train.py` (5 WandB sites) | per-quadrant fan-out: 4 occupancy keys + 0–4 rabbit-quadrant keys | | |
| `tests/environment/test_per_quadrant_and_per_rabbit_info.py` | new (T1–T4) | | |

**Conclusion**: [one-line summary]

---

## Open questions for the user (raised by senior-developer pre-implementation)

These are not blocking the developer; the plan above commits to a default
answer for each. Surfaced for visibility:

1. **Quadrant tie-breaker on odd grid sizes.** Plan uses `>= height//2` so a
   center cell on a 9×9 grid goes BR, not TL. The hypervigilance configs are
   all 10×10, so this is academic for Round 2 / Round 3. Alternative: round to
   nearest, or split center cells across two quadrants — neither was
   requested. Default chosen: `>=`. *Override only if a future config relies
   on a different rule.*

2. **`Episode/MeanDistRabbit_<Q>` definition for multi-rabbit-per-quadrant
   configs.** Plan averages per-instance distances within a quadrant
   (`np.mean(means_per_neutral[mask_q])`). Alternative: report `min` (nearest
   rabbit in that quadrant) — closer to the aggregated `MeanDistRabbit`'s
   `jnp.min` semantics, but harder to interpret across windows. Default
   chosen: mean. *The hypervigilance configs all have 1 rabbit per quadrant;
   default and alternative are equivalent for Round 2. Decide the convention
   before any future multi-rabbit-per-quadrant config lands.*

3. **`quadrant_idx` dtype in Dreamer replay buffer.** Plan casts to `float32`
   for buffer-dtype homogeneity (matching how `hit_predator` etc. are cast
   per `dreamer_v3_trainer.py:609-633`). The cast is reversed at numpy
   aggregation time via `.astype(np.int32)`. Alternative: keep `int32`
   throughout and special-case the buffer dtype check. Default chosen: cast.
   *Functionally equivalent; cast is the lower-friction option.*

4. **No new YAML keys vs. exposing `quadrant_split` as configurable.** Plan
   bakes `>= height//2` directly into the env step. Alternative: add a config
   key (e.g., `environment.quadrant_split: 'half'` vs `'thirds'`). The
   project's `no fallback defaults` rule means a configurable would need to
   be `config.get_mandatory(...)`, breaking every existing config. Default
   chosen: hard-code. *If quadrant geometry ever needs to vary across
   experiments, design a separate plan.*
