---
title: "Basic difficulty curriculum — forage-vs-predator, 5 sparse levels"
topic: basic_curriculum
status: active
created: 2026-06-17
last_updated: 2026-06-17
phase: 1
develop_link: ../../../develop/active/refactors/CONFIG_LAYERING_AND_EXPERIMENT_REORG.md
---

# Basic difficulty curriculum — forage-vs-predator, 5 sparse levels

## Purpose (plain language)

This document defines a **5-step difficulty ladder** for the grid-world agent. Each
step ("level") is a small world the agent must survive in, and the worlds get
progressively harder as you climb the ladder. The curriculum is deliberately
**stripped down to one skill at a time**: every world contains only **food** (which
the agent must eat to stay alive) and, from level 1 onward, one or more **predators**
(which chase and injure the agent). There is no sensory noise and no visual clutter —
no rabbits, no hiding bushes, no rocks. The point is to study foraging-under-threat in
the cleanest possible setting before layering on harder perception challenges later.

The ladder grows along **two axes at once**: the **world gets bigger** (a cramped 5×5
grid → a roomy 10×10 grid) and the **predators get more dangerous** (start absent, then
slow and short-sighted, then fast, then numerous, then numerous *and* far-seeing while
food becomes scarce). The five levels are:

- **Level 0** — empty 5×5 world, food only, no predator. Pure "find and eat food".
- **Level 1** — same small world, plus one **slow, short-sighted** predator. First "avoid the threat".
- **Level 2** — bigger 8×8 world, one **fast** predator. Adds speed and space.
- **Level 3** — full 10×10 world, **two fast** predators. Multiple simultaneous threats.
- **Level 4** — 10×10 world, two **far-seeing** predators, **scarce** food. Hardest: nowhere is safe.

These configs are the **first real users** of the project's new
[config-layering feature](../../../develop/active/refactors/CONFIG_LAYERING_AND_EXPERIMENT_REORG.md):
each level is a short YAML that says "start from the standard environment, then change
only these few things". This document records what each level is, why it sits where it
does on the ladder, and the validation evidence that all five worlds load and build
correctly.

> **Scope note.** This is a **scene-design document**, not a hypothesis-test. It defines
> the curriculum worlds and verifies they instantiate as intended. It does **not**
> pre-register a confirm/refute outcome, choose an agent architecture, or commit a
> training horizon — those belong to whichever training study consumes this ladder. The
> Launch Manifest below is therefore left as a template for the consuming study to fill.

---

## 1. What "basic" means here (design rationale)

The user locked the following decisions; this doc implements them, it does not re-debate them:

- **Progression axis = grid-size + threat together.** Both world size and predator threat
  grow toward the 10×10 endpoint. Difficulty is not isolated to one knob — a curriculum
  that grew only the grid, or only the threat, would not match the intended "everything
  gets harder" shape.
- **Forage-vs-predator ONLY.** No perceptual noise, no visual-channel complexity. The
  noise block (`perceptual_noise`, which stays disabled) and the sensor block (`sensory`)
  are inherited unchanged from the standard environment — they are *not* touched by any
  level.
- **Clean minimal scene per level.** Each world contains exactly food + that level's
  predator(s). The standard environment's rabbits, its four hiding-predator resources,
  and its rocks/bushes are all **dropped**.

### Why these specific knobs

| Knob | What it controls | How the ladder uses it |
|---|---|---|
| Grid size (`height`/`width`) | How much room to evade and how far food is | 5×5 → 8×8 → 10×10 |
| Predator `move_interval` | Predator speed (steps between predator moves) | 3 (slow) → 1 (fast) |
| Predator `detection_range` | How far the predator can sense the agent | 3 (short) → 5 (long, "keen") |
| Predator count | Number of simultaneous threats | 0 → 1 → 2 |
| Food `count` | Foraging pressure (scarcity) | generous (2 on small grid, scaled up on big) → scarce (2 on big grid) |

All other predator fields (damage, stamina set, attack delay, smell/visual `properties`,
nociception intensity) are held **constant across every level** at the standard
environment's predator values, so the only things that change level-to-level are the five
knobs above. This keeps the difficulty progression interpretable.

---

## 2. The five levels

Grid sizes, predator settings, and food counts per level. "Adds vs previous" names the
single new difficulty ingredient introduced at each step.

| Level | File stem | Grid | Predators | Predator speed (`move_interval`) | Predator sight (`detection_range`) | Food | Obstacles | Adds vs previous |
|---|---|---|---|---|---|---|---|---|
| **0** | `00-forage_5x5` | 5×5 | 0 | — | — | 2 | 0 | Baseline: pure foraging/navigation |
| **1** | `01-slowPred_5x5` | 5×5 | 1 | 3 (slow) | 3 (short) | 2 | 0 | First predator → first avoidance |
| **2** | `02-fastPred_8x8` | 8×8 | 1 | 1 (fast) | 3 (short) | 3 | 0 | Predator speed + a larger world |
| **3** | `03-multiPred_10x10` | 10×10 | 2 | 1 (fast) | 3 (short) | 4 | 0 | Two simultaneous threats, full grid |
| **4** | `04-keenPred_10x10` | 10×10 | 2 | 1 (fast) | 5 (long) | 2 | 0 | Long-range sight + scarce food (hypervigilance) |

Predator non-varying fields, identical on every predator at every level (copied from the
standard environment's predator): `damage: [15.0, 45.0]`, `max_stamina: 30`,
`stamina_recovery_rate: 1`, `hunt_stamina_threshold: 0.7`,
`lose_interest_multiplier: 1.5`, `attack_delay: 3`, `nociception_intensity: 0.9`,
`properties: [0.0, 0.7, 0.5, 0.0, 0.0]`, `properties_std: [0.0, 0.4, 0.4, 0.0, 0.0]`.

Food is a single resource entry per level with `properties: [1.0, 0, 0, 0, 0]`,
`max_consumption: 12`, `regeneration_delay: 0`, no damage. On levels 3–4 the two
predators spawn in opposite corners (top-left, bottom-right) but patrol the whole grid,
so the agent cannot find a permanently safe quadrant.

### How layering keeps these files sparse

Each file is ~50 lines and declares only `extends: environment/default` plus the
`environment` scene block (grid, `resources`, `entities`, `location_areas`, `obstacles`).
Everything else — body physiology, sensors, the disabled noise block, behaviour
measures — is inherited from the standard environment and never restated.

**The list-replace footgun (handled).** The merge deep-merges nested dicts but **replaces
list-valued keys wholesale** — it does not element-merge lists. Three scene keys are
list-valued: `resources`, `entities`, `obstacles`. Consequences, all handled in these
configs:

- To get a **clean scene**, each level **declares its own** `resources` / `entities` /
  `obstacles`, which replaces the standard environment's versions entirely (dropping its
  rabbits, four hiding-predator resources, and rocks/bushes).
- To drop animals entirely (Level 0) or obstacles (all levels), the list is written as an
  **explicit empty list** (`entities: []`, `obstacles: []`). Omitting the key would
  silently leak the standard environment's entities/obstacles back in.
- Because list items are replaced wholesale, **every predator/food entry is fully
  specified** with all mandatory fields — you cannot partial-override one predator field
  and inherit the rest.

`behavior_measures` stays enabled via inheritance; its online counters are lightweight,
and bush/rabbit-dependent measures simply read as zero in these clean scenes.

---

## 3. Launch Manifest (template — to be filled by the consuming training study)

This curriculum defines **worlds**, not runs. A training study that adopts the ladder
owns seed count, horizon, agent config, and the tag scheme. The recommended tag pattern
is `<algo>_basic_L<level>_s<seed>` (e.g. `recurrentppo_basic_L4_s0`), all rows sharing
`wandb-group: basic_curriculum`. Fill the table below at that study's design time.

| Run | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Status | Node | GPU | Launched at | WandB run ID | Log path |
|---|---|---|---|---|---|---|---|---|---|---|---|
| _tbd_ | _tbd_ | _tbd_ | basic_curriculum | prod | _tbd_ | planned | — | — | — | — | — |

### 3.1 Configs produced (system of record for the scene definitions)

| Level | Env config path | Agent config |
|---|---|---|
| 0 | `configs/environment/experiment/basic/00-forage_5x5.yaml` | _study-owned_ |
| 1 | `configs/environment/experiment/basic/01-slowPred_5x5.yaml` | _study-owned_ |
| 2 | `configs/environment/experiment/basic/02-fastPred_8x8.yaml` | _study-owned_ |
| 3 | `configs/environment/experiment/basic/03-multiPred_10x10.yaml` | _study-owned_ |
| 4 | `configs/environment/experiment/basic/04-keenPred_10x10.yaml` | _study-owned_ |

---

## 4. Validation results

Each config was loaded through the layering resolver and built into an environment state
to confirm: (a) the `extends:` merge succeeds and every mandatory key is satisfied
post-merge, (b) the grid/predator/food/obstacle counts match the intended level, and
(c) the scene is clean (zero rabbits, zero hiding-predator resources, obstacles as
intended). Commands used (under the project conda interpreter):

```
cfg = load_env_config(path); params = load_env_params(cfg)          # extends merge + mandatory keys
state = jax_reset(params, jax.random.PRNGKey(0))                    # builds the world
# read: params.height/width, len(params.predator_indices),
#       len(params.neutral_indices), state.res_pos.shape[0], state.obs_pos.shape[0]
```

| Level | Grid | Predators | Rabbits (neutral) | Hiding-predator resources | Food | Obstacles | Result |
|---|---|---|---|---|---|---|---|
| 0 `00-forage_5x5` | 5×5 | 0 | 0 | 0 | 2 | 0 | OK |
| 1 `01-slowPred_5x5` | 5×5 | 1 | 0 | 0 | 2 | 0 | OK |
| 2 `02-fastPred_8x8` | 8×8 | 1 | 0 | 0 | 3 | 0 | OK |
| 3 `03-multiPred_10x10` | 10×10 | 2 | 0 | 0 | 4 | 0 | OK |
| 4 `04-keenPred_10x10` | 10×10 | 2 | 0 | 0 | 2 | 0 | OK |

All five worlds load, build, and match their intended shape. Rabbits and
hiding-predator resources are zero everywhere (clean scene confirmed), and obstacles are
zero everywhere (all levels declare `obstacles: []`). Food resources appear as one entry
expanded to the configured `count`.

### Schema constraints resolved during authoring

- **Unified `entities:` schema, not the legacy split.** The standard environment uses the
  modern unified `entities:` list (one entry per animal with `class` + `behaviour`), not
  the older `predators:` / `neutral_animals:` lists that the archived basic configs used.
  These new levels use `entities:` to stay consistent with the base they extend (mixing
  schemas triggers a deprecation warning and the unified list wins).
- **A `behaviour: hunt` predator requires all five chase fields.** `detection_range`,
  `max_stamina`, `stamina_recovery_rate`, `hunt_stamina_threshold`, and
  `lose_interest_multiplier` are mandatory for hunt entities — each predator entry
  declares all five (plus `move_interval`, `damage`, `attack_delay`,
  `nociception_intensity`, `spawn_area`, `patrol_area`, `properties`, `properties_std`).
- **Chemical-property vector width must match the base.** The standard environment's
  animals use a 5-element `properties` vector; each predator here uses the matching
  5-element vector so the per-entity arrays stack without a width mismatch.
- **Areas are 1-indexed inclusive and clamped to the grid.** All `spawn_area`,
  `patrol_area`, and grass `location_area` values are written within each level's grid
  (e.g. `[[1,1],[5,5]]` for 5×5, `[[1,1],[10,10]]` for 10×10). `start_pos` is the grid
  centre per level (`[3,3]` on 5×5, `[4,4]` on 8×8, `[5,5]` on 10×10) and fits each grid.
  Note: the loader 0-indexes the 1-indexed YAML `start_pos`, so e.g. YAML `[3,3]` reads
  internally as `[2,2]` — both refer to the same centre cell.

---

## 5. Links

- Config-layering feature these configs are the first real consumer of:
  [CONFIG_LAYERING_AND_EXPERIMENT_REORG.md](../../../develop/active/refactors/CONFIG_LAYERING_AND_EXPERIMENT_REORG.md)
- Archived original basic configs (legacy, full, reference only):
  `configs/environment/experiment/archive/basic/`
- Standard environment (the base every level extends): `configs/environment/default.yaml`
