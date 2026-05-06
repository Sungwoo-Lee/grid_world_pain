---
title: Olfaction Parity (Predator vs. Neutral) and Observability Gates Verification
topic: issues
status: archive
created: 2026-05-01
last_updated: 2026-05-01
---

# Olfaction Parity (Predator vs. Neutral) and Observability Gates Verification

> **Status**: COMPLETED
> **Opened**: 2026-05-01
> **Implemented by**: Gemini
> **Date**: 2026-05-01 00:36:12
> **Related**: [09_sensors_and_observation.md](../environment/09_sensors_and_observation.md), [ISSUE_06_HIDDEN_STATES_INTEROCEPTIVE_NOCICEPTION.md](ISSUE_06_HIDDEN_STATES_INTEROCEPTIVE_NOCICEPTION.md)

---

## Context

Two correctness questions about the observation pipeline need empirical verification:

1. **Olfactory parity across entity classes.** Predators, neutral animals, resources, and obstacles all feed olfaction through the same `sense_resource()` call in [sensor.py:289-295](../../src/environment/sensor.py#L289-L295), and the four chemical contributions are simply summed. The decay law and active-mask are identical. We want a black-box experimental confirmation that **a single neutral animal** with property vector $P$ at position $X$ produces the **same olfactory observation** as **a single predator** with the same property $P$ at the same position $X$, holding everything else fixed. This protects against subtle divergences (e.g., property-sampling differences, masking, std defaults) that could leak entity identity into the chemical channel.

2. **Observability gates.** [default.yaml:283-284](../../configs/environment/default.yaml#L283-L284) sets `injury_observable: false` and `nutrition_observable: false`. The gating in [sensor.py:269-275](../../src/environment/sensor.py#L269-L275) and [sensor.py:324-329](../../src/environment/sensor.py#L324-L329) excludes these slots from both the observation vector and the breakdown. We want a black-box test that confirms **no Injury or Nutrition slot appears anywhere in the observation** (including no zero-padded placeholder), and that toggling the flags to `true` re-introduces exactly one slot each in the documented position.

Both checks are quick to run and produce regression-quality artifacts (CSV/JSON of observations) usable by future PR reviews.

## Analysis

### Olfaction code path

[sensor.py:5-22](../../src/environment/sensor.py#L5-L22) defines:

```python
def sense_resource(agent_pos, res_pos, res_active, res_property, radius, decay_power):
    diff = res_pos - agent_pos
    dist = jnp.linalg.norm(diff, axis=-1)
    decay = jnp.where(dist < 0.001, 2.0, 1.0 / (jnp.power(dist, decay_power) + 1e-10))
    mask = jnp.logical_and(res_active, dist <= radius)
    weighted_props = res_property * decay[:, None] * mask[:, None]
    return jnp.sum(weighted_props, axis=0)
```

The function is class-agnostic — it does not look at entity type. The four chemical contributions in [sensor.py:291-295](../../src/environment/sensor.py#L291-L295) are computed with the same `sensor_radius` and `sensor_decay`, then summed. Therefore, **for identical (position, property, active=True) inputs, predator and neutral contributions must be bit-equal**.

Possible failure points to probe:
- `pred_property_sampled` vs. `neutral_property_sampled`: both sampled by `_sample_property` in [core.py:750-753](../../src/environment/core.py#L750-L753) with `properties_std`. If `properties_std=0`, sampled = mean. The test must use `properties_std: [0,0,0,0,0]` to make sampling deterministic.
- `res_active` is supplied for resources only; predators/neutral pass `jnp.ones(...)` literally — confirm both are length-1 ones in the single-entity case.
- Position equality: spawn areas must be a single cell so both entities are forced to a known coordinate; or, rely on identical `(spawn_area, count=1, seed)` and read back `state.pred_pos` / `state.neutral_pos` to confirm.

### Observability gates code path

[sensor.py:269-275](../../src/environment/sensor.py#L269-L275) builds `obs_parts` conditionally:

```python
if params.injury_observable:
    obs_parts.append(jnp.array([state.injury_level / params.max_injury]))
if params.nutrition_observable:
    obs_parts.append(jnp.array([state.nutrition / params.max_nutrition]))
```

[sensor.py:320-360](../../src/environment/sensor.py#L320-L360) does the matching omission in `get_observation_breakdown`. Configuration flows through [config_loader.py:284-285](../../src/environment/config_loader.py#L284-L285) → [config_loader.py:399-400](../../src/environment/config_loader.py#L399-L400) → [state.py:161-162](../../src/environment/state.py#L161-L162). When both flags are `false`, the observation should start with `Satiation` (one scalar), and breakdown keys should not contain `"Injury"` or `"Nutrition"`.

Possible failure points to probe:
- Silent zero-padding in the assembler (currently absent — verify it stays absent).
- Renderer / `build_sensory_viz` ([sensor.py:362-400](../../src/environment/sensor.py#L362-L400)) consuming an offset that assumes the gated slots exist.
- Perceptual-noise clip arrays (`noise_clip_min`, `noise_clip_max`) being indexed by gated slot — if `apply_perceptual_noise` ([sensor.py:255-259](../../src/environment/sensor.py#L255-L259)) iterates `breakdown` and uses `modality_map[name]`, the gated keys are skipped naturally; confirm.

## Implementation Plan

### Design

Two **standalone verification scripts** under `scripts/verification/`, each producing:
- a console-printed pass/fail summary (`assert`-driven),
- a `tmp/<datetime>_<name>.json` log with raw observation tensors and metadata for archival.

No source code is modified — these are read-only black-box tests that load minimal YAML configs, call `jax_reset` + `get_observation`, and compare slices.

The plan also adds two **minimal YAML configs** (one per scenario) so tests are reproducible and not coupled to `default.yaml`.

#### Olfaction parity: experiment design

Two minimal 5×5 grids, identical except for entity class:

| Variant | Resources | Predators | Neutrals | Obstacles |
|---|---|---|---|---|
| A — Neutral-only | 0 | 0 | 1 (property `P`, std 0) | 0 |
| B — Predator-only | 0 | 1 (property `P`, std 0) | 0 | 0 |

Constraints:
- `random_start_pos: false`, `start_pos: [2, 2]` (agent fixed at center).
- Single 1×1 `spawn_area` for the entity, e.g. `[[0,0],[0,0]]`, so its placement is deterministic.
- `properties_std: [0,0,0,0,0]` → property sampling is identity.
- Same `sensor_radius`, `sensor_decay`, `vector_size=5`, same seed.
- All other sensors disabled or configured identically; `perceptual_noise.enabled: false`.

Procedure:
1. Load Variant A → `params_A`, `state_A = jax_reset(params_A, key)`, `obs_A = get_observation(state_A, params_A, apply_noise=False)`.
2. Load Variant B → `params_B`, `state_B = jax_reset(params_B, key)`, `obs_B = get_observation(state_B, params_B, apply_noise=False)`.
3. Assert `state_A.neutral_pos[0] == state_B.pred_pos[0]` (sanity — entity actually placed at same cell).
4. Locate the `Olfaction` slice using `get_observation_breakdown(params_A)`; assert it matches the slice from `params_B` (same start index, same length).
5. Assert `jnp.allclose(obs_A[olf_slice], obs_B[olf_slice], atol=1e-6)`. Print both vectors.
6. Repeat for **at least 3 distinct property vectors** (`P1=[0,0.5,0.7,0,0]`, `P2=[1,0,0,0,0]`, `P3=[0.3,0.3,0.3,0.3,0.3]`) and **3 agent-entity distances** (Manhattan d ∈ {1, 2, 4}) by varying `start_pos` while keeping the entity cell fixed.

Output: a 9-row table (3 properties × 3 distances) of `obs_A_olf`, `obs_B_olf`, `delta = obs_A − obs_B`. Pass criterion: max abs delta < 1e-6 across all rows.

#### Observability gates: experiment design

Four scenarios, all on `default.yaml`-derived minimal configs:

| Scenario | `injury_observable` | `nutrition_observable` | Expected delta (vs. base) |
|---|---|---|---|
| S1 | `false` | `false` | obs starts at Satiation; breakdown lacks both keys |
| S2 | `true`  | `false` | obs starts at Injury; breakdown contains `"Injury"` only |
| S3 | `false` | `true`  | obs starts at Nutrition; breakdown contains `"Nutrition"` only |
| S4 | `true`  | `true`  | obs starts at Injury → Nutrition; both keys present |

Procedure (per scenario):
1. Load scenario config → `params`, `state = jax_reset(params, key)`.
2. Compute `breakdown = get_observation_breakdown(params)`; assert key membership matches the table.
3. Compute `obs = get_observation(state, params, apply_noise=False)`.
4. Assert `obs.shape[0] == sum(breakdown.values())` (no silent padding).
5. For S2/S3/S4, set a known non-trivial `injury_level` / `nutrition` (via `state.replace(...)`) and read back the gated slot — confirm value equals `level / max_*`.
6. For S1, set the same `injury_level`/`nutrition` and confirm `obs.shape` is unchanged and no slot reflects them.

Pass criterion: all four scenarios match the expected breakdown and shape; numerical reads in S2/S3/S4 are within 1e-6 of the analytical values.

### File Changes

> Reminder: per CLAUDE.md, Claude does **not** implement. The implementing agent should create the files below.

#### `configs/verification/olfaction_parity_neutral.yaml` (new file)

Minimal config with one neutral animal, no predators/resources/obstacles, deterministic placement, no perceptual noise. Derived from `default.yaml` with the following overrides:

```yaml
environment:
  height: 5
  width: 5
  start_pos: [2, 2]
  random_start_pos: false
  max_steps: 10
  rest_action_enabled: true
  eat_action_enabled: false
  resources: []           # no food
  predator_enabled: false
  predators: []
  neutral_animals:
    - name: "test_neutral"
      count: 1
      properties: [0.0, 0.5, 0.7, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 0
      nociception_intensity: 0.0
      spawn_area: [[0, 0], [0, 0]]
      patrol_area: [[0, 0], [0, 0]]
  obstacles: []
  location_areas:
    - type: "grass"
      area: [[0, 0], [4, 4]]

body:                     # copy from default.yaml unchanged
  ...
sensory:
  using_sensory: true
  olfactory_enabled: true
  sensor_radius: 20
  vector_size: 5
  decay_power: 2.0
  collision_sensor_range: 1
  location_sensor: false
  nociception_enabled: false
  visual_sensor_enabled: false
  proprioception_enabled: false
  injury_observable: true
  nutrition_observable: true
  interoceptive_nociception_enabled: false
  interoceptive_convolution_enabled: false
  interoceptive_kernel_tau: 3.0
  interoceptive_kernel_length: 12

perceptual_noise:
  enabled: false
  modalities: { ... }     # copy from default.yaml unchanged (parser requires it)
```

#### `configs/verification/olfaction_parity_predator.yaml` (new file)

Identical to the neutral file above, except the entity block:

```yaml
  predator_enabled: true
  predators:
    - name: "test_predator"
      count: 1
      properties: [0.0, 0.5, 0.7, 0.0, 0.0]   # SAME as neutral
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 0
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
      spawn_area: [[0, 0], [0, 0]]
      patrol_area: [[0, 0], [0, 0]]
      detection_range: 0
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 1.0
      attack_delay: 999
      lose_interest_multiplier: 1.0
  neutral_animals: []
```

`detection_range: 0`, `attack_delay: 999`, and `move_interval: 0` keep the predator inert at step 0 so its position can't drift before the observation is captured.

#### `configs/verification/observability_gates_S1.yaml` ... `S4.yaml` (4 new files)

Copy of `default.yaml` with two lines varied per scenario (`sensory.injury_observable`, `sensory.nutrition_observable`) and `perceptual_noise.enabled: false`. Keep the rest of the config intact so the test exercises a realistic observation vector.

#### `scripts/verification/check_olfaction_parity.py` (new file)

```python
"""Verifies that neutral-animal and predator olfactory observations are equal
when their (position, property) inputs are equal. Black-box; reads two YAMLs."""
import json
import os
import sys
import datetime as dt
import jax
import jax.numpy as jnp
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset
from src.environment.sensor import get_observation, get_observation_breakdown


def load(path):
    with open(path) as f:
        return Config(yaml.safe_load(f))


def olf_slice(breakdown):
    idx = 0
    for name, dim in breakdown.items():
        if name == "Olfaction":
            return slice(idx, idx + dim)
        idx += dim
    raise KeyError("Olfaction not in breakdown")


def run_one(cfg_path, seed, agent_pos_override=None):
    params = load_env_params(load(cfg_path))
    if agent_pos_override is not None:
        params = params.replace(start_pos=jnp.array(agent_pos_override))
    state = jax_reset(params, jax.random.PRNGKey(seed))
    obs = get_observation(state, params, apply_noise=False)
    bk = get_observation_breakdown(params)
    return params, state, obs, bk


def main():
    seed = 0
    cases = []
    for prop_name, prop in [("P1", [0, 0.5, 0.7, 0, 0]),
                            ("P2", [1, 0, 0, 0, 0]),
                            ("P3", [0.3]*5)]:
        for d, agent in [("d1", [0, 1]), ("d2", [0, 2]), ("d4", [0, 4])]:
            # The implementing agent should programmatically rewrite the
            # `properties` field in a temp YAML, OR pre-generate 3×3 = 9 configs
            # under configs/verification/parity/<prop>_<d>/. Either is fine.
            ...

    # Per case, run neutral + predator variants, slice Olfaction, assert close.
    rows = []
    for ... :
        _, state_a, obs_a, bk_a = run_one(neutral_cfg, seed, agent)
        _, state_b, obs_b, bk_b = run_one(predator_cfg, seed, agent)
        assert (state_a.neutral_pos[0] == state_b.pred_pos[0]).all()
        s_a, s_b = olf_slice(bk_a), olf_slice(bk_b)
        assert s_a == s_b, (s_a, s_b)
        delta = jnp.max(jnp.abs(obs_a[s_a] - obs_b[s_b]))
        rows.append({"prop": prop_name, "dist": d,
                     "obs_neutral": obs_a[s_a].tolist(),
                     "obs_predator": obs_b[s_b].tolist(),
                     "max_abs_delta": float(delta)})
        assert delta < 1e-6, rows[-1]

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"tmp/{ts}_olfaction_parity.json"
    os.makedirs("tmp", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"PASS — {len(rows)} cases, max delta {max(r['max_abs_delta'] for r in rows):.2e}")
    print(f"Log: {out}")


if __name__ == "__main__":
    main()
```

#### `scripts/verification/check_observability_gates.py` (new file)

```python
"""Verifies injury_observable / nutrition_observable correctly include / exclude
slots in get_observation and get_observation_breakdown."""
import json
import os
import sys
import datetime as dt
import jax
import jax.numpy as jnp
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset
from src.environment.sensor import get_observation, get_observation_breakdown


def main():
    expectations = {
        "S1": {"injury": False, "nutrition": False, "must_have": [],
               "must_not_have": ["Injury", "Nutrition"]},
        "S2": {"injury": True,  "nutrition": False, "must_have": ["Injury"],
               "must_not_have": ["Nutrition"]},
        "S3": {"injury": False, "nutrition": True,  "must_have": ["Nutrition"],
               "must_not_have": ["Injury"]},
        "S4": {"injury": True,  "nutrition": True,  "must_have": ["Injury", "Nutrition"],
               "must_not_have": []},
    }

    rows = []
    for tag, exp in expectations.items():
        cfg = Config(yaml.safe_load(open(f"configs/verification/observability_gates_{tag}.yaml")))
        params = load_env_params(cfg)
        state = jax_reset(params, jax.random.PRNGKey(0))
        # Force known interoceptive values to read them back via the slot.
        state = state.replace(injury_level=jnp.float32(40.0),
                              nutrition=jnp.float32(70.0))
        obs = get_observation(state, params, apply_noise=False)
        bk = get_observation_breakdown(params)

        for k in exp["must_have"]:     assert k in bk, (tag, bk)
        for k in exp["must_not_have"]: assert k not in bk, (tag, bk)
        assert obs.shape[0] == sum(bk.values()), (tag, obs.shape, bk)

        # Numerical read for present slots.
        idx = 0
        slot_values = {}
        for name, dim in bk.items():
            if name in ("Injury", "Nutrition"):
                slot_values[name] = float(obs[idx])
            idx += dim
        if "Injury" in bk:
            assert abs(slot_values["Injury"] - 40.0/params.max_injury) < 1e-6, slot_values
        if "Nutrition" in bk:
            assert abs(slot_values["Nutrition"] - 70.0/params.max_nutrition) < 1e-6, slot_values

        rows.append({"scenario": tag, "obs_dim": int(obs.shape[0]),
                     "breakdown": {k: int(v) for k, v in bk.items()},
                     "slot_values": slot_values})

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("tmp", exist_ok=True)
    out = f"tmp/{ts}_observability_gates.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("PASS — all 4 scenarios match expectations")
    for r in rows:
        print(f"  {r['scenario']}: dim={r['obs_dim']}  keys={list(r['breakdown'])}")
    print(f"Log: {out}")


if __name__ == "__main__":
    main()
```

## Checkpoints

The implementing agent should verify the following during implementation:

- [x] Both YAMLs in `configs/verification/parity/*` parse via `load_env_params` without warnings (`properties` not `property`). [00:37:01]
- [x] `state.neutral_pos[0]` in the neutral variant equals `state.pred_pos[0]` in the predator variant (proves the placement is identical, not just by-construction). [00:38:20]
- [x] Olfaction breakdown dimension equals `params.vector_size` (5) in both variants. [00:38:20]
- [x] `obs_A[olf] == obs_B[olf]` to ≤1e-6 across all 9 (property × distance) cases. [00:38:20]
- [x] For S1, `obs.shape[0] == sum(breakdown.values())` and `"Injury" not in breakdown` and `"Nutrition" not in breakdown`. [00:39:00]
- [x] For S4, the first two observation slots, after dividing by `max_injury` / `max_nutrition`, recover the injected `injury_level` / `nutrition` values exactly. [00:39:00]
- [x] Rerun both scripts with `apply_noise=True` once at the end as a smoke test. [00:40:00]

## Implementation Report

> **Implemented by**: Gemini
> **Date**: 2026-05-01 00:40:00

- Created verification configs and scripts as planned.
- Discovered and fixed missing mandatory configuration keys in minimal YAMLs (`visual_sensor_range`, `nociception_size`, `local_view_size`).
- Fixed coordinate indexing (1-based in YAML, 0-based in script overrides).
- Verified Olfaction parity across 9 cases (3 properties x 3 distances) with 0.0 bit-parity.
- Verified observability gates across 4 scenarios (S1-S4) with correct slot inclusion/exclusion and numerical accuracy.
- Smoke test with `apply_noise=True` confirmed that the parity test still passes when using identical noise seeds (since `apply_noise` is deterministic given a key and state).


## Verification Report

> **Verified by**: Claude
> **Date**: 2026-05-01

### Re-execution evidence

Both scripts were re-run from a clean shell to confirm Gemini's reported results.

**`check_olfaction_parity.py`** → `PASS — 9 cases, max delta 0.00e+00`
Log: `tmp/20260501_004213_olfaction_parity.json`. All 9 (3 properties × 3 distances) cases produce **bit-identical** neutral vs. predator olfactory slices (delta = 0.0, not just within 1e-6). Sample row: at d=1 (`agent_pos=[0,1]`, entity at `[0,0]`), both observations equal `[0, 0.5, 0.7, 0, 0]` (decay = 1, full property recovered). At d=2, both equal `[0, 0.125, 0.175, 0, 0]` = property × 0.25 = property × (1/d²). Non-vacuous: properties P1/P2/P3 all produce non-zero, distance-modulated outputs.

**`check_observability_gates.py`** → `PASS — all 4 scenarios match expectations`
Log: `tmp/20260501_004158_observability_gates.json`.
- S1 (both `false`): obs_dim=27, breakdown has neither `Injury` nor `Nutrition`.
- S2 (`injury=true`): obs_dim=28, `Injury` slot = 0.4 (= 40/max_injury=100 ✓).
- S3 (`nutrition=true`): obs_dim=28, `Nutrition` slot = 0.7 (= 70/max_nutrition=100 ✓).
- S4 (both `true`): obs_dim=29, both slots present and numerically correct.
For all four, `obs.shape[0] == sum(breakdown.values())` — no silent zero-padding.

### File-level verification

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `configs/verification/olfaction_parity_neutral.yaml` | new | ✅ | Loads cleanly; `properties_std=0` ensures deterministic property sampling. |
| `configs/verification/olfaction_parity_predator.yaml` | new | ✅ | Mirrors neutral config; `attack_delay=999`, `move_interval=0` keep predator inert at t=0. |
| `configs/verification/observability_gates_S1.yaml` | new | ✅ | `injury_observable: false`, `nutrition_observable: false` — confirmed via grep + script. |
| `configs/verification/observability_gates_S2.yaml` | new | ✅ | `injury=true`, `nutrition=false`. |
| `configs/verification/observability_gates_S3.yaml` | new | ✅ | `injury=false`, `nutrition=true`. |
| `configs/verification/observability_gates_S4.yaml` | new | ✅ | Both `true`. |
| `scripts/verification/check_olfaction_parity.py` | new | ✅ | Note: in-place override via `cfg.to_dict()` works because `Config.to_dict()` ([src/utils/config.py:37-38](../../src/utils/config.py#L37-L38)) returns the live `_config` reference; mutation propagates to `load_env_params(cfg)`. Bit-parity (delta=0.0) for all 9 cases proves the override is taking effect. |
| `scripts/verification/check_observability_gates.py` | new | ✅ | `state.replace(injury_level=…, nutrition=…)` numerical reads match analytical values to 1e-6. |

### Out-of-scope changes

- ⚠️ `configs/environment/default.yaml` is currently dirty (`56 +/60 −`). **This is pre-existing user work**, not Gemini's: the session-start `gitStatus` already listed `M configs/environment/default.yaml` *before* the plan was written. No action required.

### Implementation Report deviations

Gemini noted three minimal-config fixes during implementation:
1. Added missing mandatory keys (`visual_sensor_range`, `nociception_size`, `local_view_size`) — confirmed present in YAMLs; `get_mandatory()` would otherwise raise.
2. Adjusted YAML 1-based / script 0-based coordinate handling — `state.neutral_pos[0] == [0,0]` is asserted in the script and passes, so the convention is internally consistent.
3. Added a smoke test with `apply_noise=True` (logged in checkpoints).

None of these deviate from intent.

**Conclusion**: ✅ Both verification objectives empirically confirmed — predator/neutral olfactory observations are **bit-equal** for matched (position, property) inputs (max delta = 0.0 across 9 cases), and the `injury_observable`/`nutrition_observable` gates correctly include/exclude slots with no padding across all 4 scenarios.
