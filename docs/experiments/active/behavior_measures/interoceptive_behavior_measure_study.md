---
title: Interoceptive Behavior-Measure Study — finding measures of foraging & avoidance vs. nutrition & injury
topic: behavior_measures
status: active
created: 2026-06-29
last_updated: 2026-06-29
aliases: [behavior-measure-study, interoceptive-measure-study]
---

# Interoceptive Behavior-Measure Study

## Purpose (read this first)

This is a long-running study with **one goal**: discover the right *measures* of an agent's
behavior — specifically **foraging** (going to food and eating) and **avoidance** (keeping away
from a threat) — and how those behaviors change with the agent's **internal state**: how hungry
it is (**nutrition**, 0 = starving … 100 = full) and how hurt it is (**injury**, 0 = unhurt …
100 = near death).

The key word is *measures*. We do **not** know in advance which number best captures "the agent
forages more when hungry" or "the agent avoids more when injured". **Finding that measure is the
deliverable, not an input.** So this study must not start by assuming a metric.

### How we work (the method — non-negotiable)

1. **Trajectory-first (qualitative).** Build a small probe environment, run the frozen agent in
   it, and *watch what it actually does* step-by-step (via the `trajectory-story` skill, which
   reads the `.rec.gz` recordings that `scripts/eval_rollout.py --record` writes). Look at the
   behavior before naming any number.
2. **Metric-second (quantitative).** Only once a trajectory pattern *clearly and repeatably*
   tracks the internal-state sweep do we crystallize it into an episode-level number. The metric
   is then **grounded in observed behavior**, not assumed up front.

This order matters: a metric chosen before looking can measure the wrong thing confidently.

### The agent we probe (fixed for the whole study)

One frozen checkpoint, never retrained:

- **Run:** `20260627-015427_rppo_basic05_randinit_n112`
- **Checkpoint:** `models/8900007` (~8.9M steps, the final one)
- **Why this one:** it was trained with **fully randomized** starting states — nutrition drawn
  uniformly from [0, 100] and injury from [0, 100] — on a 10×10 grid with food, a hunting
  predator (0–2 of them), and a harmless wandering "rabbit" (0–2). Because its training covered
  the *entire* [0,100]×[0,100] internal-state square, **every** start-state we test is
  in-distribution. That removes the trap that derailed an earlier study, where a model trained
  only from full-nutrition/zero-injury produced off-distribution wandering that masqueraded as
  "hypervigilance" (see [[20260624_0517_indist_random_init_reverses_hypervig]]).
- **Olfaction:** smell decay power = 2.0 (steep). Probe configs must match this (it is the
  default, so no override is needed).
- **Grid:** trained on 10×10 — probes default to 10×10 to stay in-distribution, though grid size
  is a free dial (a robust agent should be size-invariant).

## Plan (phases)

We isolate **one factor at a time**, simplest first. Each phase runs the trajectory-first loop
above before any metric is declared.

| Phase | Vary | Hold fixed | Question being probed |
|---|---|---|---|
| **1a** | nutrition 0→100 | injury 0, no threat, food at a **fixed** spot | Does hunger change foraging? (controlled, comparable trajectories) |
| **1b** | nutrition 0→100 | injury 0, no threat, food **sparse/random** | Does the Phase-1a signal survive a naturalistic layout? |
| **2**  | injury 0→100 | (threat present and/or food present — design TBD after Phase 1) | Does injury change avoidance and/or foraging? |
| later | nutrition × injury | — | Build toward a 2D map once single-factor measures are validated |

Phase 1a vs 1b choice is recorded: **controlled first to find the signal, naturalistic to
confirm it generalizes** (user decision, 2026-06-29).

Phase 2's exact target (avoidance vs. foraging-suppression vs. both) is **deliberately left open**
— it is itself part of what the trajectory-first observation in Phase 1 will inform.

## Current status

- **2026-06-29** — Study scoped. Subject model chosen (`basic05_randinit_n112`, ckpt 8900007),
  method fixed (trajectory-first → emergent metrics), Phase 1a stimulus design chosen
  (controlled, then naturalistic). Anchor doc created. **Next: build Phase 1a configs, run the
  nutrition sweep, watch trajectories.**

## How to run one probe (operational reference)

```
# 1. roll out the frozen agent in a probe env, recording trajectories
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/eval_rollout.py \
  --config environment/experiment/behavior_probes/<probe>.yaml \
  --checkpoint results/JAX_RecurrentPPO/20260627-015427_rppo_basic05_randinit_n112/models/8900007 \
  --record
#   (omit --agent_config: it auto-loads the model's saved config.yaml)

# 2. read the trajectories step-by-step (qualitative, FIRST)
#    -> use the `trajectory-story` skill on the written .rec.gz files

# 3. (optional, canonical video) render to mp4
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/render_recordings.py <recordings_dir>
```

## Links

- Direction / perspective: [[experiment_environment_design_perspective]]
- Earlier concrete probe designs (conflict/hypervigilance): [[experiment_environment_designs_v1]]
- Why the in-distribution model matters: [[20260624_0517_indist_random_init_reverses_hypervig]]
- Phase 1a configs: `configs/environment/experiment/behavior_probes/core/forage_nutrition/`

## Findings log

### 2026-06-29 — Phase 1a first run (single seed=42, 1 ep/level, nutrition 0/25/50/75/100)

**Setup:** 10×10, agent fixed start, single food 6 cells straight ahead ("Down"), no threat,
injury 0. Subject ckpt 8900007. Recordings: `results/eval/forage_nutrition/nutr{000..100}/`.

**Qualitative result — hunger changes the *timing* of foraging, not the path.** Every survivor
walks the identical straight 6-step line to the food (no meandering). What scales with starting
nutrition:

| nutrition | pre-departure Rest steps | step reaches food | at-food behavior |
|---|---|---|---|
| 0 | — | **dies @ step 2** (starves crossing) | n/a — survival floor, not foraging |
| 25 | ~0 | t8 | one long continuous eat-bout to ~65, then leaves |
| 50 | 2 | t10 | continuous eat to ~96 |
| 75 | 2 | t11 | eat to 100 then rest (sated) |
| 100 | 4 | t13 | nibble-and-rest top-up, holds ~100 |

**Emergent candidate measures (trajectory-grounded, not assumed):**
- **(A) Latency to reach food** = steps until first on-food-cell. Monotonic ↑ with nutrition (8→10→11→13).
- **(B) Pre-departure delay** = Rest/non-approach steps before committing to the food. ↑ with satiety (0→2→2→4).
- **(C) Eat-bout structure** = hungry → one long refill bout; full → eat-rest top-up oscillation. (Harder to scalarize.)

**Caveats / next:**
- Single seed, single episode per level — the monotonic trend needs **multiple seeds** to trust (action sampling is stochastic; geometry is fixed).
- **Nutrition 0 is a death floor.** Useful foraging range ≈ 15–100. The most urgent-forager regime (just above the floor, e.g. 10/15/20) is unsampled — add a **finer low-end sweep**, possibly with food closer so a starving agent can reach it and we can watch desperation foraging.
- First action is always "Eat" regardless of level — likely an init artifact, ignore.
- **Next step:** finer low-end + multi-seed to confirm measures (A)/(B) are monotonic, then crystallize one as the Phase-1a metric.

**Status:** Phase 1a controlled sweep RUN; signal found (satiety delays foraging). Not yet
multi-seed-confirmed. Phase 1b (naturalistic) not started.

### 2026-06-29 — Phase 1a quantitative confirmation (8 levels × 8 seeds)

Levels nutr 0/10/15/20/25/50/75/100, seeds 42–49. Recordings:
`results/eval/forage_nutrition_ms/nutr{level}/`. Measures computed from `snapshots`:
**A = latency to reach food** (first step nutrition rises = first successful eat);
**B = departure delay** (first step the agent leaves the start cell).

| nutr | survived | reach-food A | departure-delay B |
|---|---|---|---|
| 0  | 0/8 | never | 1 |
| 10 | 0/8 | never | 1 |
| 15 | 0/8 | never | 1 |
| 20 | 8/8 | 13 | 1 |
| 25 | 8/8 | **9** | 1 |
| 50 | 8/8 | 11 | 3 |
| 75 | 8/8 | 11 | 3 |
| 100 | 8/8 | 14 | 5 |

**Key results:**
1. **The probe is deterministic** — every measure is ±0.0 across all 8 seeds (fixed geometry, no
   animals → no behavioral variance). So **1 episode/level suffices**; seeds add nothing here.
   (Multi-seed will matter again only once we add random placement or threats.)
2. **Sharp survival floor between nutr 15 and 20** for food 6 cells away: ≤15 starves en route
   (0/8), ≥20 survives (8/8). The floor is a controllable boundary (moves with food distance).
3. **Two distinct foraging-disruption signatures emerge — opposite ends, different causes:**
   - **Satiety dithering (high nutrition):** the agent sits and *Rests* before bothering to
     forage. Captured cleanly by **B (departure delay): monotonic 1→3→5** with satiety. This is
     the clean "hunger drives foraging *initiative*" measure.
   - **Desperation meander (near the death floor, nutr20):** the agent leaves immediately (B=1)
     but takes a *disorganized, meandering path* (reaches @13 vs 25's @9) — reproducible across
     all 8 seeds. Inflates A at the low end. NOT a satiety-delay; a path-efficiency disruption.
   - Net: **A (reach latency) is U-shaped** (compound of both effects); **B is the clean,
     monotonic satiety measure.**

**Crystallized Phase-1a metric candidate:** **pre-departure delay B** = Rest/non-approach steps
before the agent commits to moving toward food. Monotonic in satiety, deterministic, trajectory-
grounded. A second measure — **path inefficiency** (reach-steps minus shortest-path) — captures
the separate near-floor desperation-meander effect and is worth keeping as a distinct readout.

**Status:** Phase 1a COMPLETE. Signal found and confirmed; metric candidate (B) crystallized;
desperation-meander noted as a second, distinct signature. **Next: Phase 1b (naturalistic /
random food placement) to test whether B and the meander survive outside the fixed geometry.**
