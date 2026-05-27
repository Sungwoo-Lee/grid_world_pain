---
title: "Branch Comparison: `feature/fixSensorFlag` vs `feature/tuningEnv`"
topic: meta
status: archive
created: 2026-03-04
last_updated: 2026-04-12
---

# Branch Comparison: `feature/fixSensorFlag` vs `feature/tuningEnv`

> **Date**: 2026-02-20  
> **Current Branch**: `feature/fixSensorFlag`  
> **Reference Branch**: `feature/tuningEnv`  
> **Total**: 25 files changed, +1,433 / −1,821 lines (net −388 lines — current branch has more code)

---

## Motivation: Training Speed Regression

The `feature/tuningEnv` branch introduced multiple environment enhancements (multi-predator, dynamic sensors, obstacle types, etc.) but suffered a **~2x training speed regression** — from **~1.9s/it** (Phase 2 baseline) to **~3.9s/it**.

The `feature/fixSensorFlag` branch was created to **isolate the exact cause** of this regression by porting features from `tuningEnv` one at a time, measuring speed after each step:

| Sub-Phase | Feature Ported | Speed | Verdict |
|-----------|---------------|-------|---------|
| Phase 3-1 | Damage Sampling | ~2.0s/it | ✅ No impact |
| Phase 3-2 | Vectorized Resource Respawn | ~1.97s/it | ✅ No impact |
| Phase 3-3 | Multi-Predator Support | ~1.97s/it | ✅ No impact |
| Phase 3-4 | Dynamic Sensor Assembly | ~1.98s/it | ✅ No impact |
| Phase 3-5 | Optimized Visual Channels (matmul) | ~1.93s/it | ✅ **Faster** |

### Root Cause: Visual Sensor Implementation

The regression was caused by the **visual sensor** (`sense_visual` in `sensor.py`):

- **`tuningEnv` (slow)**: Per-entity-type `jax.vmap` contributions with `lax.scan` over obstacles, producing a **dynamic channel count** (7 + N obstacle types). This created a massive JIT compilation graph.
- **`fixSensorFlag` (fast)**: Single `jnp.matmul` over a unified entity property matrix with **fixed 8 channels**. GPU processes all entities and cells in one parallel pass.

```python
# fixSensorFlag approach — single matmul, ~1.93s/it
matches = (cell_coords[:, None] == all_pos[None, :])           # [Cells, Entities]
vis_entities = jnp.matmul(matches.astype(jnp.float32), all_props)  # [Cells, 8]
```

> [!IMPORTANT]
> The `sensor.py` difference is the **single most critical divergence** between branches. Any future sync must preserve the matmul approach.

---

## Development Phases

- **Phase 1**: Zero-impact infrastructure & metadata updates.
- **Phase 2**: Neuromodulation model & training logic updates.
- **Phase 3** (3-1 through 3-5): Incremental porting with performance isolation (see table above).
- **Phase 4**: Professional telemetry renderer & evaluation pipeline.

---

## Summary Table

| File | Category | Status | Lines Changed |
|------|----------|--------|---------------|
| `renderer.py` | Renderer | ⚠️ Minor diff | +37/−0 |
| `grid_world.py` | Renderer | ✅ New file (not in tuningEnv) | +725 |
| `evaluation_core.py` | Evaluation | ⚠️ Major diff | +382 structural |
| `core.py` | Environment | ⚠️ Major diff | +230 |
| `sensor.py` | Environment | ⚠️ Major diff | +215 |
| `config_loader.py` | Environment | ⚠️ Major diff | 242 lines changed |
| `state.py` | Environment | ⚠️ Minor diff | 6 lines (comments) |
| `train.py` | Training | ⚠️ Major diff | +293 |
| `evaluation.py` | Evaluation | ⚠️ Moderate diff | +69 |
| `main.py` | Utility | ⚠️ Minor diff | 6 lines |
| `neuromodulator.py` | Models | ⚠️ Major diff | +282 |
| `recurrent_ppo_network.py` | Models | ⚠️ Moderate diff | +29 |
| `recurrent_ppo_trainer.py` | Models | ✅ Minor diff | 4 lines |
| `dreamer_v3_nnx.py` | Models | ⚠️ Major diff | +250 |
| `dreamer_v3_trainer.py` | Models | ⚠️ Major diff | +270 |
| `drqn_trainer.py` | Models | ⚠️ Minor diff | 16 lines |
| `modulated_layer_norm_gru_cell.py` | Models | 🔴 Deleted in tuningEnv | 84 lines |
| `neuromodulated_dreamer_v3.yaml` | Config | 🔴 Deleted in tuningEnv | 49 lines |
| `neuromodulated_ppo.yaml` | Config | ⚠️ Moderate diff | 14 lines |
| `visualization.yaml` | Config | ⚠️ Minor diff | 5 lines |
| `command.sh` | Utility | 🔴 Deleted in tuningEnv | 12 lines |
| `.gitignore` | Utility | ✅ Minor diff | 3 lines |
| `assets/agent_food.jpg` | Asset | ✅ New file | binary |
| `assets/assets.pptx` | Asset | ✅ New file | binary |
| `docs/IMPORTANT_ISSUES.md` | Docs | ✅ New file | 31 lines |

---

## Detailed File-by-File Analysis

### 1. Renderer (`src/environment/renderer.py`)

**Status**: ⚠️ Nearly synced — 2 remaining differences

| Aspect | `fixSensorFlag` | `tuningEnv` |
|--------|-----------------|-------------|
| `render_jax_state` signature | Has `info=None` parameter | No `info` parameter |
| Acute Damage pod (right panel) | ✅ Present (shows damage breakdown) | ❌ Not present |
| `calculate_drive` import | ✅ Imported | ❌ Not imported |
| `draw_categorical_visual` | ✅ Synced (labels param, color rotation) | ✅ Same |
| `COLORS` dict | ✅ Synced | ✅ Same |
| Obstacle icon lookup | ✅ Synced (`params.obstacle_names`) | ✅ Same |

**Remaining diff**: The Acute Damage pod in the right panel and `info` parameter are Phase 4 features intentionally added to `fixSensorFlag`. These are **forward additions** not yet in `tuningEnv`.

---

### 2. Grid World (`src/environment/grid_world.py`)

**Status**: ✅ New file — only exists in `fixSensorFlag`

This is a **duplicate renderer** maintained for backward compatibility. It mirrors `renderer.py` functionality including all Phase 4 updates.

---

### 3. Evaluation Core (`src/utils/evaluation_core.py`)

**Status**: ⚠️ Major structural difference

| Aspect | `fixSensorFlag` | `tuningEnv` |
|--------|-----------------|-------------|
| Stats recording | Per-step `jax.device_get` | Batched GPU→CPU transfer via stacked arrays |
| Stats output | `pandas DataFrame.to_csv` | `csv.writer` with entity positions |
| Stats filename | `ep_{N}_stats.csv` | `{N:06d}ep_stats.csv` |
| Stat columns | Compact metrics + damage breakdown | Full entity positions + raw observations |
| Stat headers | Hardcoded column names | Dynamically generated from sensor breakdown |
| `info` passing to renderer | ✅ Passes `info` dict | ❌ Does not pass `info` |
| `main()` guard | ❌ Not present | ✅ Has `if __name__` guard |
| Sensor labels | `['NEU', 'RCK']` order | Same |

**Key insight**: `tuningEnv` uses a high-performance batched approach that minimizes GPU→CPU synchronization, while `fixSensorFlag` uses a simpler per-step approach with pandas. The `tuningEnv` stats CSV format is more detailed (includes full entity positions per step).

---

### 4. Core Environment (`src/environment/core.py`)

**Status**: ⚠️ Major difference — 230 lines

| Aspect | `fixSensorFlag` | `tuningEnv` |
|--------|-----------------|-------------|
| Entity placement (`_place_entities`) | ✅ Has collision-aware placement via `jax.lax.scan` | ❌ Not present |
| Spawn sampling (`_sample_unoccupied`) | ✅ Has retry-based placement | ❌ Not present |
| Resource respawn (`_respawn_resources`) | ✅ Has ghost-aware respawn logic | ❌ Not present |
| `update_body` nutrition refill | Multi-line | Compressed to single `ate_food_gain` variable |
| Collision damage targeting | Placed **before** predator update | Placed **after** predator update |
| Resource respawn in `jax_step` | Simple `vmap` (no collision avoidance) | Same simple `vmap` |

**Key insight**: `fixSensorFlag` has extensive collision-aware entity placement functions (`_place_entities`, `_sample_unoccupied`, `_respawn_resources`) that are **not used in the hot path** (`jax_step`) — they exist for `jax_reset` only. `tuningEnv` doesn't have these functions, suggesting reset placement is handled differently or these were added as Phase 5 preparation.

The `attempted_pos` calculation for collision damage targeting is the same logic but placed at a different point in the function — `fixSensorFlag` calculates it early (before predator update), `tuningEnv` calculates it later.

---

### 5. Sensor (`src/environment/sensor.py`)

**Status**: ⚠️ Major difference — 215 lines

| Aspect | `fixSensorFlag` | `tuningEnv` |
|--------|-----------------|-------------|
| Visual sensor (`sense_visual`) | **8 fixed channels** with matmul optimization | **Dynamic channels** (7 + N obstacle types) |
| Channel mapping | 0-7 fixed (Grass→Neutral) | 0-6 fixed + 7..N for obstacle types |
| Entity contributions | Single matmul with concatenated property matrix | Per-entity-type `jax.vmap` contributions |
| Obstacle visual channels | All obstacles → channel 6 ("Rock") | Each obstacle type → unique channel (7+) |
| Noise clipping | Inline within `apply_perceptual_noise` | Separate list-based accumulation |
| `clip_min`/`clip_max` vectors | Built inline at the end | Built in the loop as lists |

**Key insight**: `tuningEnv` has a **dynamic visual channel count** that scales with obstacle types (e.g., "rock" and "tree" get separate channels 7 and 8), while `fixSensorFlag` uses a **fixed 8-channel** matmul approach where all obstacles share channel 6. The `fixSensorFlag` approach is faster (single matmul) but less expressive.

---

### 6. Config Loader (`src/environment/config_loader.py`)

**Status**: ⚠️ Major difference — 242 lines

| Aspect | `fixSensorFlag` | `tuningEnv` |
|--------|-----------------|-------------|
| Helper function | Uses `strict_get(obj, key, label)` utility | Uses inline `p_get`/`obs_get`/`n_get` lambdas |
| Predator loading order | Checks `predator_enabled` first, then loads list | Always loads list first, then checks `predator_enabled` |
| Damage range parsing | Uses `to_range()` helper | Inline `[d, d]` fallback |
| Obstacle name tuple | `obstacle_names_tuple` variable | `obstacle_names` variable |
| Empty obstacle fallback | `obstacle_names_tuple = ()` | `obstacle_names = ("rock",)` |
| Default values | No defaults (strict) | Some defaults (e.g., `blocking=True`, `nociception=0.3`) |

---

### 7. State (`src/environment/state.py`)

**Status**: ⚠️ Minor — comment-level differences only

| Field | `fixSensorFlag` | `tuningEnv` |
|-------|-----------------|-------------|
| `res_damage` | `# [num_res]` | `# [num_res, 2] [min, max]` |
| `pred_damage` | `# [num_pred]` | `# [num_pred, 2] [min, max]` |
| `obs_damage` | `# [num_obs]` | `# [num_obs, 2] [min, max]` |

> Note: The actual data shapes are identical — only the docstring comments differ.

---

### 8. Training Script (`train.py`)

**Status**: ⚠️ Major — 293 lines extra in `fixSensorFlag`

| Aspect | `fixSensorFlag` | `tuningEnv` |
|--------|-----------------|-------------|
| `DebugTimer` class | ✅ Has 50-line timing utility | ❌ Not present |
| `parse_jax_trace()` | ✅ Has 40-line profiler trace parser | ❌ Not present |
| Timer checkpoints | `timer.start/stop` calls throughout | No timing |
| Device selection | Verbose with `"cude"` typo handling | Simplified, no typo handling |
| Config print section | Wrapped with timer | No timer |

---

### 9. Evaluation Script (`evaluation.py`)

**Status**: ⚠️ Moderate — 69 lines extra in `fixSensorFlag`

| Aspect | `fixSensorFlag` | `tuningEnv` |
|--------|-----------------|-------------|
| Device selection | Verbose with `"cude"` typo handling | Simplified |
| Observation spec logging | ✅ Detailed breakdown print | ❌ Not present |
| `get_observation_breakdown` import | ✅ Imported | ❌ Not imported |
| Visualization config merge | Separate merge step | Omitted (falls through to defaults) |

---

### 10. Neuromodulator (`src/models/neuromodulator.py`)

**Status**: ⚠️ Major — 282 lines extra in `fixSensorFlag`

| Aspect | `fixSensorFlag` | `tuningEnv` |
|--------|-----------------|-------------|
| `ModulatorOutput` fields | 4 fields: `z_percept`, `z_percept_add`, `z_memory`, `temperature` | 3 fields: `z_percept`, `z_memory`, `temperature` |
| Modulation types | Supports "Multiplicative" AND "PreActivation" | Only "Multiplicative" |
| `DreamerNeuromodulatorRNN` | ✅ Full Dreamer-specific modulator class | ❌ Not present |
| `head_percept_add` | ✅ Ferguson-style threshold shift head | ❌ Removed |
| `modulation_type` attribute | ✅ Present | ❌ Removed |
| `modulated_layer_norm_gru_cell.py` | ✅ Exists (84 lines) | 🔴 Deleted |

---

### 11. RecurrentPPO Network (`src/models/recurrent_ppo_network.py`)

**Status**: ⚠️ Moderate — 29 lines

| Aspect | `fixSensorFlag` | `tuningEnv` |
|--------|-----------------|-------------|
| Modulation style routing | `if modulation_type == "PreActivation": ... else:` | Always multiplicative gating |
| `percept_add_bias_init` | ✅ Passed to modulator | ❌ Removed |
| `modulation_type` | ✅ Stored in `self` | ❌ Removed |

---

### 12. Neuromodulated PPO Config (`configs/models/ppo/neuromodulated_ppo.yaml`)

**Status**: ⚠️ Simplified in tuningEnv

| Field | `fixSensorFlag` | `tuningEnv` |
|-------|-----------------|-------------|
| Modulation type comments | Extended (Multiplicative, PreActivation, null) | Single line |
| `percept_add_bias_init` | ✅ `0.0` | ❌ Removed |
| `percept_bias_init` comment | "Gain head (gamma)" | "Perceptual gate" |

---

### 13. Visualization Config (`configs/visualization/visualization.yaml`)

| Field | `fixSensorFlag` | `tuningEnv` |
|-------|-----------------|-------------|
| `fps` | `5` | `2` |
| `video_dpi` | `300` | `300` (but has `100` commented) |
| `tree` icon mapping | ✅ Present | ❌ Removed |

---

### 14. Files Only in `fixSensorFlag` (Not in `tuningEnv`)

| File | Purpose |
|------|---------|
| `src/environment/grid_world.py` | Duplicate renderer (725 lines) |
| `src/models/modulated_layer_norm_gru_cell.py` | LayerNorm GRU cell for modulation (84 lines) |
| `configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml` | DreamerV3 neuro-modulated config (49 lines) |
| `command.sh` | Shell utility script (12 lines) |
| `assets/agent_food.jpg` | Composite icon |
| `assets/assets.pptx` | Asset source file |
| `docs/IMPORTANT_ISSUES.md` | Known issues documentation |

---

## Guiding Principle

> [!CAUTION]
> **If a difference between branches is NOT related to training speed, sync `fixSensorFlag` to match `feature/tuningEnv` as closely as possible.** The only reason to diverge is if the `tuningEnv` approach causes the ~2x speed regression. All other differences should be resolved to match `tuningEnv`.

---

## Classification of All Differences

Each difference is classified as either:
- 🔒 **KEEP** — Divergence exists specifically for training speed. Do NOT sync.
- 🔄 **SYNC** — Not speed-related. Must be updated to match `tuningEnv`.
- ➕ **FORWARD** — New feature added in `fixSensorFlag` that `tuningEnv` doesn't have yet (acceptable to keep).

| # | File | Difference | Class | Reason |
|---|------|-----------|-------|--------|
| 1 | `sensor.py` | Fixed 8-ch matmul vs dynamic N-ch vmap | 🔒 KEEP | **Root cause of regression** |
| 2 | `sensor.py` | Noise clipping style | 🔄 SYNC | Cosmetic, no speed impact |
| 3 | `core.py` | Entity placement functions | 🔒 KEEP | Overlap checking caused speed regression; handle last |
| 4 | `core.py` | `attempted_pos` ordering | 🔄 SYNC | Logic order, no speed impact |
| 5 | `core.py` | `update_body` nutrition style | 🔄 SYNC | Cosmetic, no speed impact |
| 6 | `config_loader.py` | `strict_get` vs inline helpers | 🔄 SYNC | Code style |
| 7 | `config_loader.py` | Predator loading order | 🔄 SYNC | Logic order |
| 8 | `config_loader.py` | Empty obstacle fallback | 🔄 SYNC | Default value |
| 9 | `state.py` | Damage array comments | 🔄 SYNC | Comments only |
| 10 | `train.py` | `DebugTimer` + `parse_jax_trace` | 🔒 KEEP | Useful for future profiling |
| 11 | `train.py` | Device selection verbosity | 🔄 SYNC | Code simplification |
| 12 | `evaluation.py` | Device selection / obs spec | 🔄 SYNC | Code simplification |
| 13 | `evaluation.py` | Vis config merge step | 🔄 SYNC | Unnecessary step |
| 14 | `evaluation_core.py` | Stats format (pandas vs csv) | 🔄 SYNC | Evaluation only |
| 15 | `evaluation_core.py` | Stats filename | 🔄 SYNC | Naming convention |
| 16 | `evaluation_core.py` | `main()` guard | 🔄 SYNC | Best practice |
| 17 | `renderer.py` | `info` + Acute Damage pod | ➕ FORWARD | Phase 4 enhancement |
| 18 | `neuromodulator.py` | PreActivation + DreamerV3 mod | ➕ FORWARD | Architecture extension |
| 19 | `recurrent_ppo_network.py` | Modulation routing | ➕ FORWARD | Required by #18 |
| 20 | `neuromodulated_ppo.yaml` | Extended config | ➕ FORWARD | Required by #18 |
| 21 | `visualization.yaml` | fps, `tree` icon | 🔄 SYNC | Config preference |
| 22 | `modulated_layer_norm_gru_cell.py` | New file | ➕ FORWARD | Architecture extension |
| 23 | `neuromodulated_dreamer_v3.yaml` | New file | ➕ FORWARD | Architecture addition |
| 24 | `grid_world.py` | Duplicate renderer | ➕ FORWARD | Backward compat |

---

## Remaining Update Plan

### Phase 5A: Script Cleanup (`train.py`, `evaluation.py`) ✅

- [x] Simplify device selection in `train.py` to match tuningEnv (keep `DebugTimer` and `parse_jax_trace`)
- [x] Simplify device selection in `evaluation.py` to match tuningEnv
- [x] Remove observation spec logging block and `get_observation_breakdown` import from `evaluation.py`
- [x] Remove visualization config merge step from `evaluation.py`

### Phase 5B: Environment Logic Sync (`core.py`, `config_loader.py`, `state.py`) ✅

- [x] Move `attempted_pos` calculation to after predator update in `core.py`
- [x] Compress `update_body` nutrition refill to `ate_food_gain` style in `core.py`
- [x] Replace `strict_get`/`to_range` with inline helpers in `config_loader.py`
- [x] Restructure predator loading order in `config_loader.py`
- [x] Set empty obstacle fallback to `("rock",)` in `config_loader.py`
- [x] Add default values where tuningEnv has them in `config_loader.py`
- [x] Update damage array comments in `state.py`

### Phase 5C: Evaluation Core Sync (`evaluation_core.py`) ✅

- [x] Port batched GPU→CPU transfer approach from tuningEnv
- [x] Switch stats format to `csv.writer` with entity positions
- [x] Change filename convention to `{N:06d}ep_stats.csv`
- [x] Generate stat headers dynamically from sensor breakdown
- [x] Add `if __name__ == "__main__": main()` guard

### Phase 5D: Sensor Noise Cleanup (`sensor.py`) ✅

- [x] Refactor noise clipping to match tuningEnv inline approach (**keep matmul vision unchanged**)

### Phase 5E: Config Sync (`visualization.yaml`) ✅

- [x] Set `fps: 5`, add `tree` icon mapping

---

### Phase 5F: Entity Overlap Checking (`core.py` — LAST) ✅

> [!NOTE]
> **Conclusion: No overlap checking at reset — matches Craftax pattern.**
>
> Investigated Craftax's entity placement: mobs start **inactive** (`mask=False`) and are spawned
> during gameplay via `mob_map` occupancy grid + `jax.random.choice`. They never check overlaps at reset.
>
> We tested a Craftax-style occupancy grid implementation (`_place_entities_grid` with `jax.lax.scan`
> + `jax.random.choice` per entity), but it caused **significant speed regression** due to
> `jax.random.choice` with probability arrays being expensive when JIT-compiled and vmapped × 256 envs.
>
> **Final approach**: Independent `vmap` placement per entity type using `jax.random.randint` within
> spawn areas. On a 20×20 grid with ~15 entities, overlap probability is <4% — a negligible edge case
> that the step-level interaction logic handles gracefully.

- [x] Evaluate collision-aware entity placement approach for `jax_reset`
- [x] Benchmark training speed — occupancy grid caused regression, reverted to vmap
- [x] Final implementation: fast independent vmap, no overlap checking (Craftax pattern)

---

## Current `jax_reset` Implementation

Each entity type is placed **independently and in parallel** using `jax.vmap`:

```
Agent  →  random position (or fixed start_pos)
                    ↓ (no occupancy tracking)
Resources  →  vmap(jax.random.randint) within per-resource spawn_area
Predators  →  vmap(jax.random.randint) within per-predator spawn_area  
Obstacles  →  vmap(jax.random.randint) within per-obstacle spawn_area
Neutrals   →  vmap(jax.random.randint) within per-neutral spawn_area
Body state →  random or fixed nutrition/injury
```

**Key properties**:
- **O(1) per entity** — no sequential dependencies, fully parallelizable
- **No `lax.scan`** — avoids the JIT compilation overhead that caused regression
- **Spawn area constraints preserved** — each entity samples within its configured `spawn_area`
- **Rare overlaps tolerated** — <4% chance on 20×20 grid, handled by step logic

---

## Items NOT to Sync (🔒 KEEP / ➕ FORWARD)

| Item | Reason to keep |
|------|---------------|
| `sensor.py` matmul vision (fixed 8-ch) | 🔒 **Root cause fix** for training speed |
| `train.py` `DebugTimer` + `parse_jax_trace` | 🔒 Useful for future profiling |
| `renderer.py` Acute Damage pod + `info` | ➕ Phase 4 telemetry |
| `neuromodulator.py` extended architecture | ➕ Future experiment support |
| `recurrent_ppo_network.py` modulation routing | ➕ Required by neuromodulator |
| `modulated_layer_norm_gru_cell.py` | ➕ Architecture extension |
| `neuromodulated_dreamer_v3.yaml` | ➕ DreamerV3 config |
| `grid_world.py` duplicate renderer | ➕ Backward compatibility |

---

## Neuromodulation Algorithm Comparison

> **Date**: 2026-02-20
> **Reference Document**: `docs/NEUROMODULATION_ALGORITHM.md`

This section provides a detailed comparison of the neuromodulation implementations across both branches. The `feature/tuningEnv` branch implements the **full architecture** described in `NEUROMODULATION_ALGORITHM.md`, while `feature/fixSensorFlag` has a **simplified subset**.

### High-Level Summary

| Capability | `fixSensorFlag` (current) | `tuningEnv` |
|------------|--------------------------|-------------|
| **PPO Neuromodulation** | ✅ Multiplicative only | ✅ Multiplicative + PreActivation |
| **DreamerV3 Neuromodulation** | ❌ Not implemented | ✅ Fully implemented |
| **Perceptual Modulation Modes** | 1 (Multiplicative) | 2 (Multiplicative + PreActivation) |
| **ModulatorOutput Fields** | 3 (`z_percept`, `z_memory`, `temperature`) | 4 (`z_percept`, `z_percept_add`, `z_memory`, `temperature`) |
| **Injection Sites (PPO)** | A (Percept), B (Memory), C (Temperature) | A (Percept), B (Memory), C (Temperature) |
| **Injection Sites (DreamerV3)** | None | A (Percept), B (Memory), C (Reward) |
| **ModulatedLayerNormGRUCell** | ❌ Does not exist | ✅ 84 lines — LayerNorm GRU with gate_bias |
| **DreamerV3 Config** | ❌ Does not exist | ✅ `neuromodulated_dreamer_v3.yaml` (49 lines) |

---

### 1. `neuromodulator.py` — Modulator Architecture

#### 1A. ModulatorOutput (PPO)

| Field | `fixSensorFlag` | `tuningEnv` |
|-------|-----------------|-------------|
| `z_percept` | ✅ Perceptual gate signal | ✅ Perceptual gain signal (gamma) |
| `z_percept_add` | ❌ Not present | ✅ Perceptual additive signal (beta, threshold shift) |
| `z_memory` | ✅ Memory gate-bias | ✅ Memory gate-bias |
| `temperature` | ✅ Bounded scalar | ✅ Bounded scalar |

#### 1B. NeuromodulatorRNN (PPO Modulator)

| Aspect | `fixSensorFlag` | `tuningEnv` |
|--------|-----------------|-------------|
| `modulation_type` parameter | ❌ Not accepted | ✅ Accepts `"Multiplicative"` or `"PreActivation"` |
| `percept_add_bias_init` parameter | ❌ Not accepted | ✅ Bias init for beta head (default 0.0) |
| `head_percept` (gamma) | ✅ Linear → spatial grouping | ✅ Identical |
| `head_percept_add` (beta) | ❌ Not constructed | ✅ Constructed when `type = "PreActivation"` |
| `head_memory` | ✅ Linear → spatial grouping | ✅ Identical |
| `head_action` (temperature) | ✅ Softplus + clip | ✅ Identical |
| `z_perc_add_baseline` | ❌ Not present | ✅ Per-neuron learned baseline for beta |
| `z_percept_add` output | ❌ Not in NamedTuple | ✅ Returns beta signal (zeros if Multiplicative) |
| Docstring terminology | "Perceptual gate" | "Perceptual gain (gamma)" (matches algorithm doc) |

#### 1C. DreamerNeuromodulatorRNN (DreamerV3 Modulator)

| Aspect | `fixSensorFlag` | `tuningEnv` |
|--------|-----------------|-------------|
| Class existence | ❌ **Does not exist** | ✅ Full implementation (~150 lines) |
| `DreamerModulatorOutput` | ❌ Not defined | ✅ 4 fields: `z_percept`, `z_percept_add`, `z_memory`, `z_reward` |
| Dual input projections | ❌ N/A | ✅ `proj_obs` (obs mode) + `proj_imagine` (imagination mode) |
| `forward_obs()` | ❌ N/A | ✅ Observation mode — all heads active |
| `forward_imagine()` | ❌ N/A | ✅ Imagination mode — memory + reward active, percept zeroed |
| `_compute_heads()` | ❌ N/A | ✅ Shared head computation with `include_percept` flag |
| `head_reward` | ❌ N/A | ✅ Sigmoid-bounded reward interpretation (bias init +2.0) |
| `set_imagine_input_dim()` | ❌ N/A | ✅ Lazy init for imagination projection (feat_dim depends on RSSM config) |
| Per-head target dims | ❌ N/A | ✅ `embed_dim` for percept, `deter_dim` for memory, 1 for reward |
| Spatial grouping per head | ❌ N/A | ✅ `num_groups_percept` and `num_groups_memory` computed separately |

---

### 2. `recurrent_ppo_network.py` — PPO Injection Points

| Aspect | `fixSensorFlag` | `tuningEnv` |
|--------|-----------------|-------------|
| `modulation_type` attribute | ❌ Not stored | ✅ Stored as `self.modulation_type` |
| **Injection A routing** | Multiplicative only | Multiplicative vs PreActivation conditional |
| Multiplicative equation | `relu(Wx+b) * sigmoid(z_percept)` | `relu(Wx+b) * sigmoid(z_percept)` (identical) |
| PreActivation equation | ❌ Not implemented | ✅ `relu(gamma * (Wx+b) + beta)` |
| `percept_add_bias_init` passed to modulator | ❌ Not passed | ✅ Passed from config |
| `modulation_type` passed to modulator | ❌ Not passed (Multiplicative implicit) | ✅ Passed as `mod_type` |
| Injection B (Memory) | ✅ `gate_bias=mod_output.z_memory` | ✅ Identical |
| Injection C (Temperature) | ✅ `logits / mod_output.temperature` | ✅ Identical |

#### PreActivation Mode (Ferguson & Cardin Style) — tuningEnv Only

The `tuningEnv` branch implements the full pre-activation modulation from `NEUROMODULATION_ALGORITHM.md` §3A:

```python
# gamma = sigmoid(z_percept) → multiplicative neural gain (Shine et al., 2021)
# beta  = z_percept_add      → additive threshold shift (Ferguson & Cardin, 2020)
x_linear = self.input_proj(x)           # Raw pre-activation
gamma = sigmoid(mod_output.z_percept)   # Gain
beta = mod_output.z_percept_add         # Threshold shift
x_proj = relu(x_linear * gamma + beta)  # Modulated activation
```

This provides a richer affine transformation of the neuron's I/O curve compared to simple post-activation scaling:
- **Gain control** (gamma): Rescales the energy landscape (Shine et al., 2021)
- **Threshold shifting** (beta): Disinhibitory gating — positive beta lowers the activation threshold, negative raises it (Ferguson & Cardin, 2020)

---

### 3. DreamerV3 Integration — tuningEnv Only

The entire DreamerV3 neuromodulation pipeline exists **only in `tuningEnv`**. The `fixSensorFlag` branch has no modulation support in any DreamerV3 file.

#### 3A. `dreamer_v3_nnx.py` Modifications (tuningEnv)

| Component | Change |
|-----------|--------|
| `RSSM.__init__` | Accepts `modulation_enabled` flag; constructs `ModulatedLayerNormGRUCell` when True |
| `RSSM.step()` | Accepts optional `gate_bias` for Injection B |
| `RSSM.imagine_step()` | Accepts optional `gate_bias` for Injection B during planning |
| `Encoder` | Split into `body` (pre-activation MLP) and `final_act` (SiLU) for modulation injection point |
| `Encoder.forward_with_modulation()` | Supports both Multiplicative and PreActivation on encoder output |
| `WorldModel.__init__` | Accepts `modulation_config`; conditionally constructs `DreamerNeuromodulatorRNN` |
| `WorldModel` attributes | `modulation_enabled`, `modulation_type`, `modulator` |

#### 3B. `dreamer_v3_trainer.py` Modifications (tuningEnv)

| Component | Change |
|-----------|--------|
| **World Model Scan** | Carries `h_mod` alongside `prev_state`; runs `modulator.forward_obs()` per step |
| **Injection A** | `encoder.forward_with_modulation(obs, mod_output, type)` during WM training |
| **Injection B (WM)** | `rssm.step(..., gate_bias=mod_output.z_memory)` during WM training |
| **Hidden State Flow** | `h_mods_all` from WM scan → reshape to `(B*T, mod_hidden)` → `stop_gradient` → `h_mod_start` for imagination |
| **Imagination Scan** | Carries `h_mod`; runs `modulator.forward_imagine(concat(feat, action), h_mod)` per horizon step |
| **Injection B (Imag)** | `rssm.imagine_step(..., gate_bias=mod_output.z_memory)` during imagination |
| **Injection C (Reward)** | `rew = rew * mod_output.z_reward` — scales imagined reward (imagination only) |
| **get_action()** | Runs modulator in observation mode; carries `mod_h` in state dict |
| **WandB Metrics** | Logs `mod_gamma_mean/std`, `mod_memory_mean/std`, `mod_z_reward_mean`, `mod_beta_mean/std` (PreActivation) |

#### 3C. `modulated_layer_norm_gru_cell.py` (tuningEnv Only)

84-line custom Flax NNX module implementing `NEUROMODULATION_ALGORITHM.md` §5.1 for DreamerV3's RSSM:

```python
gates_ih = LN(W_ih @ x)
gates_hh = LN(W_hh @ h)
gates = gates_ih + gates_hh
reset, update, cand = split(gates, 3)
update = sigmoid(update + gate_bias)   # ← modulated when gate_bias provided
h_new = (1 - update) * h + update * tanh(cand)
```

When `gate_bias = None`, functionally identical to the original `LayerNormGRUCell`.

#### 3D. `neuromodulated_dreamer_v3.yaml` (tuningEnv Only)

Full DreamerV3 config with modulation block:
- `type: "Multiplicative"` / `"PreActivation"` / `null`
- `mod_hidden_size: 64`, `grouping_size: 1`
- `percept_bias_init: 2.0`, `percept_add_bias_init: 0.0`
- `memory_bias_init: 0.0`, `reward_bias_init: 2.0`

---

### 4. `neuromodulated_ppo.yaml` — Config Differences

| Field | `fixSensorFlag` | `tuningEnv` |
|-------|-----------------|-------------|
| `type` comment | "Modulation style (§4: ablation hook)" | Full multi-line with Multiplicative, PreActivation, null |
| `percept_bias_init` comment | "Perceptual gate bias init" | "Gain head (gamma) bias init" |
| `percept_add_bias_init` | ❌ Not present | ✅ `0.0` — threshold-shift head bias (PreActivation only) |

---

### 5. Alignment with `NEUROMODULATION_ALGORITHM.md`

| Algorithm Doc Section | `fixSensorFlag` | `tuningEnv` |
|-----------------------|-----------------|-------------|
| §2A — `NeuromodulatorRNN` (PPO) | ✅ Partial (Multiplicative only) | ✅ **Full** (Multiplicative + PreActivation) |
| §2B — `DreamerNeuromodulatorRNN` | ❌ Not implemented | ✅ **Full** (dual projections, reward head) |
| §3A — Injection A: Multiplicative | ✅ Implemented | ✅ Implemented |
| §3A — Injection A: PreActivation | ❌ Not implemented | ✅ Implemented (gamma + beta) |
| §3A — Injection B: Memory Gate-Bias | ✅ `ModulatedGRUCell` | ✅ `ModulatedGRUCell` + `ModulatedLayerNormGRUCell` |
| §3A — Injection C: Temperature (PPO) | ✅ Implemented | ✅ Implemented |
| §3B — DreamerV3 Observation Mode | ❌ Not implemented | ✅ Injections A + B |
| §3B — DreamerV3 Imagination Mode | ❌ Not implemented | ✅ Injections B + C |
| §3B — Hidden State Flow (WM → Imagination) | ❌ Not implemented | ✅ `h_mod` carry through both scans |
| §5.1 — `ModulatedGRUCell` (PPO) | ✅ Exists | ✅ Identical |
| §5.1 — `ModulatedLayerNormGRUCell` (DreamerV3) | ❌ Does not exist | ✅ 84 lines |
| §5.2 — Pass-Through Init (bias +2.0) | ✅ Percept head | ✅ Percept + reward heads |
| §5.2 — Baseline Control (`type: null`) | ✅ Works | ✅ Works |
| §5.3 — Bounded Temperature | ✅ Softplus + clip [0.1, 10.0] | ✅ Identical |
| Ablation: `modulation.type` | ✅ Multiplicative or null | ✅ Multiplicative, PreActivation, or null |
| WandB Modulator Metrics (PPO) | ✅ Logged | ✅ Logged |
| WandB Modulator Metrics (DreamerV3) | ❌ N/A | ✅ Logged (gamma, memory, z_reward, beta) |

---

### 6. What Needs Porting to Reach Full Parity

| Priority | Item | Files Affected | Effort |
|----------|------|---------------|--------|
| 🔴 High | Add PreActivation mode to PPO modulator | `neuromodulator.py`, `recurrent_ppo_network.py`, `neuromodulated_ppo.yaml` | Medium |
| 🔴 High | Port `DreamerNeuromodulatorRNN` class | `neuromodulator.py` | Medium |
| 🔴 High | Port `ModulatedLayerNormGRUCell` | New file: `src/models/modulated_layer_norm_gru_cell.py` | Easy (84 lines) |
| 🔴 High | Port DreamerV3 modulation integration | `dreamer_v3_nnx.py` (RSSM, Encoder, WorldModel) | Large |
| 🔴 High | Port DreamerV3 trainer modulation | `dreamer_v3_trainer.py` (WM scan, imagination scan, get_action) | Large |
| 🟡 Medium | Port DreamerV3 neuromod config | New file: `configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml` | Easy |
| 🟢 Low | Align terminology ("gate" → "gain/gamma") | `neuromodulator.py` docstrings | Trivial |

> [!NOTE]
> The `tuningEnv` branch is the **reference implementation** for the full neuromodulation architecture described in `NEUROMODULATION_ALGORITHM.md`. The `fixSensorFlag` branch implements only the Multiplicative PPO subset. All DreamerV3 neuromodulation and the PreActivation perceptual mode are missing from `fixSensorFlag`.

