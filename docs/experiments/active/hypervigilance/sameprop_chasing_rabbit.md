---
title: "SameProp Round 4 — chasing rabbit: does the agent flee a harmless animal that actively chases it?"
topic: hypervigilance
status: active
created: 2026-05-29
last_updated: 2026-05-29
phase: 2
aliases:
  - sameprop_chasing_rabbit
wandb_tag: "hypervigilance-round4-chasingRabbit"
develop_link: "docs/develop/active/env_entities/UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md"
supersedes: []
---

# SameProp Round 4 — chasing rabbit

## Purpose (plain English, read this first)

In our world, the agent shares a 10x10 grid with three animals that all smell
identical (the "matched-smell" / *sameProp* setup): one **predator** that can hurt
the agent, and two **rabbits** that cannot. In earlier rounds the agent learned to
treat the predator and the rabbits differently anyway — it dives for cover far more
often when the predator approaches than when a rabbit does — even though their smell
is the same. The leading guess for *how* it tells them apart is **behaviour**: the
predator actively **chases** the agent, while the rabbits just **wander**.

This round runs a simple, exploratory probe of that guess from the opposite
direction. We make the **two rabbits chase the agent** — they pursue it across the
whole map — **but they still deal zero damage**. They are a harmless animal that
*acts* like a threat. The single question we want to eyeball is: **does the agent
flee / avoid a rabbit that chases it, even though that rabbit never actually hurts
it?** If yes, that is evidence the agent reads "is-it-chasing-me" as a danger cue
rather than learning, per-animal, "this specific thing has hurt me before."

This is **exploratory** — we want to *see* the behaviour first, then decide which
avoidance measures are worth formalising. We are deliberately keeping the
pre-registration light. No hard pass/fail threshold is locked; the deliverable is a
trained run we can watch and measure, not a confirm/refute verdict.

**Why "harmless" is automatic, not a setting we tuned.** Damage in this environment
is gated by an animal's *class*, not its behaviour. Only the `predator` class is
flagged as damaging; the `neutral` class (rabbits) is harmless by construction. So
making a rabbit chase the agent cannot accidentally make it dangerous — the rabbits
stay `neutral`, keep `is_damaging = False`, and the chase is the *only* thing that
changed. (Mechanism: `ANIMAL_DAMAGING_CLASSES = {"predator"}` in the config loader;
the per-step damage gate is `at_damaging = at_animal AND animal_is_damaging` in the
environment core.)

## 1. The manipulation (vs the Round 3 baseline)

The baseline is **Round 3** — the matched-smell, food-decoupled world where the
patrolling predator's chase parameters are randomised every episode. Round 4 changes
**exactly two things** relative to Round 3:

1. **Schema port.** All animals move from the old `neutral_animals:` / `predators:`
   YAML blocks into the unified `entities:` list. This is *required*, not cosmetic:
   the old loader path always forces a `neutral_animals:` entry to wander, so a
   "hunting neutral" simply cannot be written in the legacy schema. The new
   `entities:` schema lets a `neutral`-class animal carry `behaviour: hunt`.
2. **The two rabbits now chase.** Each rabbit becomes `behaviour: hunt` with a fixed
   (non-randomised), deliberately aggressive and persistent chase profile, so the
   chasing is reliable and *visible*.

The predator is left **byte-equivalent to Round 3** (same five per-episode
randomised chase ranges). Everything else — the food layout, the four hiding
predators, obstacles, sensors, body dynamics, perceptual-noise block, the
behaviour-measure toolkit settings, episode length, and the 200-episode eval-seed
list — is carried over unchanged from Round 3.

### 1.1 Rabbit chase profile (fixed scalars)

For `behaviour: hunt`, five chase fields are mandatory. We use scalar (fixed, not
randomised) values tuned so the rabbits reliably and visibly pursue the agent:

| Field | Value | Why |
|---|---|---|
| `detection_range` | 10 | Detect the agent anywhere on the 10x10 grid — always engages. |
| `max_stamina` | 60 | Long chase budget. |
| `stamina_recovery_rate` | 1.0 | Recovers between chases. |
| `hunt_stamina_threshold` | 0.3 | Eager to re-engage the chase. |
| `lose_interest_multiplier` | 3.0 | Persistent — keeps chasing past the detection boundary. |

### 1.2 CRITICAL env gotcha — `patrol_area` clips chase movement

There is a non-obvious environment mechanic that this design has to work around. In
the environment's hunt step, an animal's new position is **clipped to its
`patrol_area` bounds even while it is actively chasing**. If we left each rabbit's
`patrol_area` set to its home quadrant (top-left or bottom-right), the rabbit could
only "chase" the agent *inside that quadrant* and could never pursue it across the
map — which would defeat the whole point of the manipulation.

So both rabbits get `patrol_area: [[1,1],[10,10]]` (the **full grid**) so they can
genuinely roam-and-chase everywhere. Their `spawn_area` stays at the original
top-left / bottom-right corners so each rabbit retains a distinct starting identity
and the `TL` / `BR` tags stay meaningful for per-animal metrics.

## 2. What to look for (informal predictions)

We are watching, not gating. The things worth eyeballing once the run trains:

- **Does the agent flee / avoid the chasing rabbits?** The headline behaviour. If
  the agent visibly puts distance between itself and a pursuing rabbit, or dives into
  cover when a rabbit closes in, that is the effect we are probing for.
- **Does the threat-distance / in-cover signal change for rabbits?** In Round 3 the
  agent's "dive into cover when a threat approaches" rate was ~88% for the predator
  vs ~51% for rabbits. If chasing pushes the rabbit number up toward the predator
  number, the chase cue matters. If it stays low despite the chase, the agent is
  using something else (e.g., a learned per-animal danger association).
- **Net displacement / path length near a chasing rabbit.** A fleeing agent should
  show larger displacement and longer paths when a rabbit is close.
- **Sanity check that rabbits never deal damage.** Injury should only ever rise from
  the predator and the static hazards, never from a rabbit contact — confirmable from
  the per-class collision/injury accounting.

These map onto metrics the behaviour-measure toolkit already logs (threat distance,
in-cover-at-threat-onset, net displacement, path length), so no new logger is needed
for a first look.

## 3. Status

Exploratory, single config, recurrent-PPO. No locked confirm/refute threshold — this
round produces a run to *observe*. If the chasing-rabbit behaviour is interesting,
a follow-up round will formalise the avoidance measures and pre-register thresholds.

## 4. Launch Manifest

| Run | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Status |
|---|---|---|---|---|---|
| R4 chasing rabbit | `recurrent_ppo_04-sameProp_R4_chasingRabbit_s42` | hypervigilance | pilot | 42 | planned |

Actual columns (Node, GPU, Launched at, WandB run ID, Log path) filled by the
training-runner at launch time.

### 4.1 Configs to produce

| Run | Env config | Agent config |
|---|---|---|
| R4 chasing rabbit | `configs/experiment/hypervigilance/04-sameProp_R4_chasingRabbit.yaml` | recurrent_ppo (existing agent config, unchanged) |

## 5. Links

- Round 3 baseline design + config this is forked from:
  [`sameprop_predator_distributional.md`](sameprop_predator_distributional.md),
  `configs/experiment/hypervigilance/03-sameProp_R3_predatorDistributional.yaml`.
- v2.0 unified-entity / per-episode-sampling refactor (the `entities:` schema used here):
  [`UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md`](../../../develop/active/env_entities/UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md).
- Behaviour-metric toolkit v1 (in-cover-at-threat-onset, threat distance, motifs):
  [`behavior_measure_toolkit_v1_design.md`](../behavior_measures/behavior_measure_toolkit_v1_design.md).

## 6. Results / Analysis / Conclusions

_(Left blank until the run trains. Returns to experiment-designer or
experiment-analyzer for fill-in.)_

## 7. Launch command (for `training-runner` reference)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/hypervigilance/04-sameProp_R4_chasingRabbit.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --episodes 10000000 \
  --num-envs 128 \
  --seed 42 \
  --device cuda:<GPU> \
  --log-interval 50 \
  --wandb-group hypervigilance \
  --wandb-job-type pilot \
  --wandb-name "recurrent_ppo_04-sameProp_R4_chasingRabbit_s42" \
  --tag "recurrent_ppo_04-sameProp_R4_chasingRabbit_s42"
```

Wrapped for the lab nodes via the standing `run_command.py` flow (never raw SSH):

```bash
./run_command.py <node> "<the train.py command above>"
```

The user picks `<node>` and `<GPU>` at hand-off time. Episode budget mirrors Round 3
(10M); for a first exploratory look the run can be stopped early once the
chasing-rabbit behaviour is observable — this is a pilot, not a thresholded prod run.
