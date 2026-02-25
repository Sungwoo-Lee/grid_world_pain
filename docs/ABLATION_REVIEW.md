# 🔬 Ablation Experiment Review

> **Last Updated**: 2026-02-25
> **Purpose**: Reconcile the original motivation of each ablation level with its current implementation, and identify settings that must be **added** or **removed** to align the two.

---

## ⚙️ How Config Loading Works

`train.py` loads configs as an **overlay** system:

1. Load `configs/environment/default.yaml` as the **base** (20x20 full environment).
2. Merge `configs/train/`, `configs/evaluation/`, `configs/logger/`, `configs/visualization/` defaults.
3. Merge the `--config` ablation YAML **on top** (deep merge).

> [!IMPORTANT]
> Any field **not specified** in the ablation YAML silently inherits from `default.yaml`. This is the root cause of most discrepancies below.

---

## 🌿 Dual-Branch Architecture

| Branch | `use_homeostatic_reward` | Reward Formula |
| :--- | :--- | :--- |
| **Survival** (Extrinsic) | `false` | `+1.0` for eating food, `-death_penalty` on death |
| **Homeostatic** (Intrinsic) | `true` | `prev_drive - curr_drive`, where `drive = √((satiation-100)² + injury²)` |

---

## 🚨 Cross-Level Issues (Inherited from Default)

These issues affect **most or all** ablation configs because the fields are not overridden:

### 1. Obstacles — Inherited 20x20 Layout
**Problem**: No ablation config specifies `environment.obstacles`. They inherit 20+ obstacles (rocks, trees, bushes) with spawn areas like `[[1, 11], [10, 20]]` — **outside a 4x4 grid**.
**Fix**: Each ablation config must add `obstacles: []` to disable them, or define 4x4-appropriate obstacles.

### 2. Neutral Animals — Inherited 20x20 Layout
**Problem**: No ablation config specifies `environment.neutral_animals`. They inherit 5 rabbits with spawn/patrol areas `[[1, 1], [20, 20]]`.
**Fix**: Add `neutral_animals: []` to all ablation configs.

### 3. Location Areas — Inherited 20x20 Layout
**Problem**: No ablation config specifies `environment.location_areas`. They inherit grass/sand areas designed for 20x20.
**Fix**: Add appropriately scaled `location_areas` or set them to cover the entire 4x4 grid.

### 4. `random_start_pos`
**Standard**: Set to `true` to ensure the agent learns generalized policies rather than memorized paths.
**Status**: Applied to all configs.

### 5. `visual_sensor_enabled` Not Disabled
**Problem**: Default has `visual_sensor_enabled: true`. No ablation config disables it. The visual sensor is active in **all** levels, adding 8 channels to the observation space even in Level 01 (Goal Only).
**Fix**: Add `visual_sensor_enabled: false` to levels where visual sensing is not part of the motivation.

### 6. `sensor_radius: 20` — Oversized for 4x4
**Problem**: Default olfactory sensor radius is 20. On a 4x4 grid, every entity is always within range.
**Fix**: Reduce to `sensor_radius: 4` for ablation configs, or leave as-is if "full knowledge" is intended.

### 7. `collision_sensor_range` — Always Active
**Problem**: `sense_collision()` is called unconditionally in `get_observation()` — it is **not** gated by any enable flag. The collision sensor is **always** part of the observation, even in levels that claim to have no sensors.
**Fix**: This is a code-level issue. For now, document that collision data is always present. The `collision_sensor_enabled` flag in configs is **not used** by the sensor code.

### 8. Interoception — Always Active
**Problem**: Interoception (satiation, nutrition, injury ratios) is always appended to the observation vector unconditionally.
**Fix**: Same as collision — document behavior. Even Level 01 receives 3 interoception channels.

### 9. `using_sensory` Flag — Not Used
**Problem**: Levels 01-03 set `using_sensory: false`, but `config_loader.py` and `get_observation()` never check this flag. Individual sensor flags (`olfactory_enabled`, etc.) are what actually control observation assembly.
**Fix**: Remove `using_sensory` from configs. Instead, explicitly set each individual sensor flag (`olfactory_enabled: false`, `nociception_enabled: false`, `proprioception_enabled: false`, `visual_sensor_enabled: false`, `location_sensor: false`).

---

## 📊 Per-Level Detail

### Level 01 — Goal Only (Survival Only)

**Motivation**: Pure pathfinding. Agent must navigate to a static food source at `[3,3]`. No threats, no body systems, no sensors. This is the simplest baseline.

| Setting | Current | Expected | Action |
| :--- | :--- | :--- | :--- |
| `resources` | 1× Food at `[3,3]` | ✅ Correct | — |
| `predators` | `[]` | ✅ Correct | — |
| `obstacles` | ❌ Inherited (20x20 rocks/trees/bushes) | `[]` | **Add** |
| `neutral_animals` | ❌ Inherited (5 rabbits) | `[]` | **Add** |
| `location_areas` | ❌ Inherited (20x20) | 4x4 coverage | **Add** |
| `random_start_pos` | ❌ Inherited (`true`) | `false` | **Add** |
| `with_satiation` | `false` | ✅ Correct | — |
| `with_injury` | `false` | ✅ Correct | — |
| `sensory.using_sensory` | `false` (ignored) | N/A | **Remove** |
| `olfactory_enabled` | ❌ Inherited (`true`) | `false` | **Add** |
| `nociception_enabled` | ❌ Inherited (`true`) | `false` | **Add** |
| `proprioception_enabled` | ❌ Inherited (`true`) | `false` | **Add** |
| `visual_sensor_enabled` | ❌ Inherited (`true`) | `false` | **Add** |

---

### Level 02 — Danger Only (Survival Only)

**Motivation**: Introduce spatial danger. Agent learns to avoid a static danger zone while surviving. Instant death on contact (no injury system). No sensors.

| Setting | Current | Expected | Action |
| :--- | :--- | :--- | :--- |
| `resources` | 1× Danger at `[3,3]` (damage: 100) | ✅ Correct | — |
| `predators` | `[]` | ✅ Correct | — |
| `obstacles` | ❌ Inherited | `[]` | **Add** |
| `neutral_animals` | ❌ Inherited | `[]` | **Add** |
| `location_areas` | ❌ Inherited | 4x4 coverage | **Add** |
| `random_start_pos` | ❌ Inherited (`true`) | `false` | **Add** |
| `with_injury` | `false` → instant death | ✅ Correct | — |
| `sensory.using_sensory` | `false` (ignored) | N/A | **Remove** |
| Individual sensors | ❌ All inherited (`true`) | All `false` | **Add** |

> [!NOTE]
> **No food source present.** The extrinsic reward (`+1.0` for eating) is never triggered. The agent's only objective is survival (avoid `-death_penalty`). Consider whether a food source should be added for reward signal.

---

### Level 03 — Predator Intro (Survival Only)

**Motivation**: Introduce a mobile threat. Agent must evade a chasing predator. Instant death on contact. No sensors, no food.

| Setting | Current | Expected | Action |
| :--- | :--- | :--- | :--- |
| `resources` | `[]` | ✅ Correct | — |
| `predators` | 1× Predator (damage: 100, instant death) | ✅ Correct | — |
| `obstacles` | ❌ Inherited | `[]` | **Add** |
| `neutral_animals` | ❌ Inherited | `[]` | **Add** |
| `location_areas` | ❌ Inherited | 4x4 coverage | **Add** |
| `random_start_pos` | ❌ Inherited (`true`) | `false` | **Add** |
| `sensory.using_sensory` | `false` (ignored) | N/A | **Remove** |
| Individual sensors | ❌ All inherited (`true`) | All `false` | **Add** |

> [!NOTE]
> **No food source present.** Same as Level 02 — no positive reward signal, only survival penalty avoidance.

---

### Level 04 — Nociception (Both Branches)

**Motivation**: First introduction of the **body system**. Injury is now a smoothed continuous state (not instant death). Agent receives a **Nociception sensor** — phasic pain signal on contact. The `rest` action is enabled for injury recovery.

| Setting | Current (Survival) | Current (Homeostatic) | Expected | Action |
| :--- | :--- | :--- | :--- | :--- |
| `resources` | `[]` | `[]` | ✅ Correct (predator-only) | — |
| `predators` | 1× (damage: 3.0) | 1× (damage: 3.0) | ✅ Correct | — |
| `eat_action_enabled` | `false` | `false` | ✅ Correct (no food) | — |
| `rest_action_enabled` | `true` | `true` | ✅ Correct | — |
| `with_injury` | `true` | `true` | ✅ Correct | — |
| `with_satiation` | `false` | `false` | ✅ Correct | — |
| `nociception_enabled` | `true` | `true` | ✅ Correct | — |
| `olfactory_enabled` | `false` | `false` | ✅ Correct | — |
| `obstacles` | ❌ Inherited | ❌ Inherited | `[]` | **Add** |
| `neutral_animals` | ❌ Inherited | ❌ Inherited | `[]` | **Add** |
| `location_areas` | ❌ Inherited | ❌ Inherited | 4x4 coverage | **Add** |
| `random_start_pos` | ❌ Inherited | ❌ Inherited | `false` | **Add** |
| `visual_sensor_enabled` | ❌ Inherited (`true`) | ❌ Inherited (`true`) | `false` | **Add** |

> [!WARNING]
> Homeostatic Level 04 has `with_satiation: false` but `use_homeostatic_reward: true`. The drive formula uses satiation: `√((satiation-100)² + injury²)`. With satiation disabled, the satiation value is frozen at the default start value (100), so `drive = √(0 + injury²) = injury`. This is **functionally correct** (drive reduces to pure injury), but should be documented.

---

### Level 05 — Olfactory (Both Branches)

**Motivation**: Introduce the **Olfactory (Chemical) sensor**. Food and danger sources now have chemical signatures. Agent must learn to follow gradients to find food and avoid danger. Satiation is enabled for the Homeostatic branch.

| Setting | Current (Survival) | Current (Homeostatic) | Expected | Action |
| :--- | :--- | :--- | :--- | :--- |
| `resources` | 1× Food + 1× Danger | 1× Food + 1× Danger | ✅ Correct | — |
| `predators` | 1× (damage: 3.0) | 1× (damage: 3.0) | ✅ Correct | — |
| `eat/rest_action_enabled` | `true`/`true` | `true`/`true` | ✅ Correct | — |
| `with_satiation` | `false` | `true` | ✅ Correct per branch | — |
| `nociception_enabled` | `true` | `true` | ✅ Correct (cumulative) | — |
| `olfactory_enabled` | `true` | `true` | ✅ Correct (new sensor) | — |
| `proprioception_enabled` | `false` | `false` | ✅ Correct | — |
| `obstacles` | ❌ Inherited | ❌ Inherited | `[]` | **Add** |
| `neutral_animals` | ❌ Inherited | ❌ Inherited | `[]` | **Add** |
| `location_areas` | ❌ Inherited | ❌ Inherited | 4x4 coverage | **Add** |
| `random_start_pos` | ❌ Inherited | ❌ Inherited | `false` | **Add** |
| `visual_sensor_enabled` | ❌ Inherited (`true`) | ❌ Inherited (`true`) | `false` | **Add** |

---

### Level 06 — Proprioception (Both Branches)

**Motivation**: Add **Proprioception sensor** (one-hot previous action) on top of Level 05. Enables temporal credit assignment — the agent knows what it just did.

| Setting | Current | Expected | Action |
| :--- | :--- | :--- | :--- |
| Sensors | Nociception ✅ + Olfactory ✅ + Proprioception ✅ | ✅ Cumulative | — |
| `obstacles` | ❌ Inherited | `[]` | **Add** |
| `neutral_animals` | ❌ Inherited | `[]` | **Add** |
| `location_areas` | ❌ Inherited | 4x4 coverage | **Add** |
| `random_start_pos` | ✅ Correct | `true` | — |
| `visual_sensor_enabled` | ❌ Inherited (`true`) | `false` | **Add** |

---

### Level 07 — Collision (Both Branches)

**Motivation**: Add **Collision sensor** (Manhattan range-1 occupancy) on top of Level 06. Agent can sense walls and blocking obstacles in adjacent cells.

| Setting | Current | Expected | Action |
| :--- | :--- | :--- | :--- |
| Sensors | +Collision ✅ (cumulative) | ✅ Correct | — |
| `collision_sensor_enabled` | `true` | ✅ Set correctly | — |
| `obstacles` | ❌ Inherited | `[]` or 4x4 obstacles | **Add** |
| `neutral_animals` | ❌ Inherited | `[]` | **Add** |
| `location_areas` | ❌ Inherited | 4x4 coverage | **Add** |
| `random_start_pos` | ✅ Correct | `true` | — |
| `visual_sensor_enabled` | ❌ Inherited (`true`) | `false` | **Add** |

> [!NOTE]
> `collision_sensor_enabled` is set but **not actually checked** by `get_observation()`. The collision sensor is always included. This flag has no effect.

---

### Level 08 — Full Homeostasis (Both Branches)

**Motivation**: The fully-featured ablation level. All sensors active, all body systems on. This is the closest to the default environment, but on a 4x4 grid.

| Setting | Current | Expected | Action |
| :--- | :--- | :--- | :--- |
| All sensors | ✅ All enabled | ✅ Correct | — |
| `location_sensor` | `true` | ✅ Correct (new addition) | — |
| `obstacles` | ❌ Inherited | `[]` or 4x4 obstacles | **Add** |
| `neutral_animals` | ❌ Inherited | `[]` | **Add** |
| `location_areas` | ❌ Inherited | 4x4 coverage | **Add** |
| `random_start_pos` | ❌ Inherited | `false` | **Add** |

> [!TIP]
> Level 08 is the only level where `visual_sensor_enabled: true` (inherited) is **intentionally correct**, since it aims for full sensory coverage.

---

## 📋 Summary of Required Changes

### Must Add to ALL Ablation Configs

```yaml
environment:
  random_start_pos: true
  obstacles: []
  neutral_animals: []
  location_areas:
    - type: "grass"
      area: [[1, 1], [4, 4]]
```

### Must Add to Levels 01–07

```yaml
sensory:
  visual_sensor_enabled: false
  visual_sensor_range: 0
```

### Must Add to Levels 01–03

Replace `using_sensory: false` with explicit individual flags:

```yaml
sensory:
  olfactory_enabled: false
  nociception_enabled: false
  proprioception_enabled: false
  collision_sensor_enabled: false
  location_sensor: false
  visual_sensor_enabled: false
  visual_sensor_range: 0
```

### Code-Level Observations (Not Config Fixes)

| Issue | Location | Note |
| :--- | :--- | :--- |
| `collision` always in observation | `sensor.py:get_observation` L274 | Not gated by any flag |
| `interoception` always in observation | `sensor.py:get_observation` L281-285 | Not gated by any flag |
| `using_sensory` unused | `config_loader.py` | Flag is never read |
| `collision_sensor_enabled` unused | `sensor.py:get_observation` | Flag exists in config but is never checked |

---

## 🔬 Sensory Progression Matrix

Shows which sensors are **intended** to be active at each level (after fixes):

| Sensor | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Collision** | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ | ✅ |
| **Interoception** | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ |
| **Nociception** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Olfactory** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Proprioception** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Visual** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Location** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

- ✅ = Intentionally enabled and controlled by config flag
- ❌ = Disabled by config flag
- ⚠️ = **Always active** regardless of config (hardcoded in `get_observation`)
