---
title: "Experiment-environment designs v1 — concrete probe worlds for the four behavior families"
topic: behavior_measures
status: planned
created: 2026-06-16
last_updated: 2026-06-16
phase: foundational
aliases: [experiment-env-designs-v1, probe-worlds-v1]
---

# Experiment-environment designs v1 — concrete probe worlds

## Purpose (read this first)

We train agents in one rich survival world, but that world is the wrong place to *read off*
what a trained agent actually does — it blends foraging, fleeing, resting, and decision-making
together, so a single average number can wash out a real behavior or invent a fake one. The
[[experiment_environment_design_perspective]] doc set the rule: to interpret a trained agent,
drop the **frozen, never-retrained** checkpoint into small purpose-built worlds, each isolating
**one** behavior, and read a simple mean-level number there — but only trust that number once a
step-by-step trajectory read confirms it.

This document is the first concrete instalment: it designs four such probe worlds, one per
behavior family the perspective doc named. In plain terms:

- **Foraging** — does the agent go get food when nothing is threatening it? (Worked here as the
  full template; the simplest case.)
- **Avoidance** — does it keep its distance from a thing that hurts it, *more* than from a
  harmless thing that moves identically?
- **Recovery** — after we injure or starve it, does it actively restore that internal state?
- **Managing conflicting needs** — when the only food sits next to the only danger, how does it
  trade off eating against staying safe?

Every design below specifies the exact small world (grid size, which entities are present or
removed), the starting internal-state dial we set to *induce* the behavior (e.g. start it hungry
to force foraging), the candidate averaged measure(s), and — mandatory — the trajectory cross-check
that must agree before the average is believed.

> **Dependency — the hungry/injured starts must be in-distribution.** These probes set the agent's
> starting nutrition/injury (e.g. hungry start = nutrition 40) to *induce* a behaviour. For that read
> to be valid the trained agent must have experienced such starts during training, which today's
> hard-coded reset bounds (nutrition upper-half only, injury lower-half only) cannot reach. The
> implementation plan that exposes the start ranges as config keys lives at
> [[CONFIGURABLE_INITIAL_STATE_RANGES]] (`docs/develop/active/refactors/`).

The single hardest lesson we carry in is from the discrimination-metric search
([[20260616_0142_discrimination_is_spatial_encounter_artifact]]): a measure that "tracks the world,
not the agent" is a **geometry artifact**, not a behavioral signal. A known *class-blind* agent
reproduced a full predator-avoidance "discrimination" gap purely because the predator roamed the
whole grid while the harmless rabbits sat confined in corners — so any agent encountered the
predator in threatening contexts more often. Remove that geometric asymmetry and the effect
vanished. **Every threat-related design below therefore carries an explicit anti-confound clause:
what would make its measure a pure geometry effect, and how the design rules that out.**

No training is launched by this doc. No YAML is written yet — the [§7 config manifest](#7-config-manifest-planned)
lists exactly which files a follow-up step would create and the precise deltas from an existing
testbed config.

---

## 0. Shared design conventions (apply to all four worlds)

These hold across every probe unless a section overrides them.

- **Frozen checkpoint, eval-only.** Each world is consumed by `scripts/eval/eval_rollout.py --config <this>`
  on an already-trained checkpoint. No weights change. The 27-dim observation is identical across all
  configs (same sensors block as the testbeds), so any checkpoint loads into any probe.
- **Confound-matched base.** All worlds inherit the cell-08 / testbed sensor, body, noise, and
  `behavior_measures` blocks byte-for-byte (the 200-seed eval list, `cue_radius: 3.0`, `obs_window: 5`,
  the M7 motif keys). Only the factor under test is edited. This is the same discipline the testbed
  search used so worlds are comparable.
- **Deterministic eval.** `eval_policy_mode: deterministic`, the locked 200-seed list, 500 max steps.
- **Multi-checkpoint, multi-seed-of-training.** A probe verdict is reported across ≥ 3 training seeds
  of the agent under study (not 3 eval seeds — eval already runs 200). Single-checkpoint reads are
  pilots only. This honors the project's multi-seed-by-default rule at the level it applies here.
- **Mean + trajectory, never mean alone.** Every measure below has a paired trajectory cross-check
  (via the `trajectory-story` skill / step dumps). The mean is provisional until the trajectory read
  tells the same story. This is perspective-doc principle #5 and the rule the discrimination search
  violated on its first pass.
- **`random_start_pos` is the default anti-confound lever.** Randomizing the agent's spawn each
  episode (already `true` in the testbeds) breaks fixed agent-vs-entity geometry; we make spawn
  geometry an explicit design variable wherever a threat is present.

---

## 1. Foraging — the worked template

### 1.1 The behavior in one sentence

When nothing threatens it, does the agent seek out and consume food to maintain its internal
energy state?

### 1.2 The isolating world — sparse food, large arena

The simplest world that exposes foraging is one with **food and no *active* threat** — but
"walk to the food you can already see" is too easy to be a real foraging read. To make the agent
actually *forage*, we use a **sparse-food** world: a **single** food source on a grid swept across
sizes, large enough at the bigger sizes that the food is usually out of perceptual range at spawn, so the agent must
**explore the world to find it** before it can approach and eat. We keep one passive danger present: a
**hiding predator** — realised in the **unified animal system** (the v2.0 refactor that merged predator
and rabbit into one programmable `Animal` class; see [[20260529_1823_unified_animal_entity_v2_0_arch]]),
**not** the legacy `type: hiding_predator` resource. Concretely it is a `class: predator`,
`behaviour: static` animal with `damage: [15, 45]` and `nociception_intensity: 0.9`: it **never moves
and never chases** (the `static` branch is pure pass-through), emits the **predator** smell and
predator sensor channel (so the agent perceives a real predator that simply lurks in place), and
inflicts contact damage when the agent shares its cell (`at_animal & animal_is_damaging` — damage is
gated on co-location, *not* on hunt state). Keeping it preserves an in-distribution predator signal
without the **chase** that would induce pursuit/avoidance dynamics and confound a foraging read. The
chasing (`behaviour: hunt`) predator and the rabbit are dropped entirely. This forks the existing
`testbed_forage_only` world by adding a single static predator-class animal, making the food sparse,
and sweeping the grid size — all free knobs (perspective DOF #1: a robust forager is size/shape-invariant).

| Knob | Value | Why |
|---|---|---|
| Grid | **swept: 5×5 (easy) · 10×10 (same as training) · 15×15 (hard)** | Three difficulties anchored to the training grid. **5×5 (easy):** the single food is almost always within `cue_radius`, so even a weak searcher succeeds. **10×10 (same as training):** the size the agent was trained on — the reference point. **15×15 (hard):** the food is rarely in range at spawn, so only a genuine explorer succeeds. Grid size is a free knob (DOF #1) — this sweep **tests** size-invariance rather than assuming it: a robust forager succeeds at every size; a fragile one succeeds at easy/same but fails at hard. |
| Hiding predator | unified-animal entity: `class: predator`, `behaviour: static`, `count: 1`, `damage: [15, 45]` (sub-lethal), `nociception_intensity: 0.9` | Stationary predator-class animal — emits the predator smell/sensor channel, **never moves or chases**, harms on co-location. Keeps an in-distribution predator signal without active threat. (Same animal system as the chasing predator; `static` needs no hunt/distributional fields.) |
| Chasing predator | removed (no `behaviour: hunt` animal) | Drop the active pursuit that would induce avoidance and confound the foraging read. |
| Rabbit (neutral) count | 0 | Remove the harmless-animal distractor entirely. |
| Food | **single source** (`count: 1`), location randomized each episode; respawns at a new random cell on consumption | Sparse food forces a genuine **search** rather than camping a known spot; randomizing the location each episode/respawn defeats memorization and any fixed spawn-to-food shortcut. (Respawn-on-consumption sustains the search demand across the whole episode — confirm the food-regeneration mechanic against the env config.) |
| Bushes | kept (`hides_agent: true`) | Obs parity + lets M7 featurize cover use even when irrelevant. |
| `random_start_pos` | true | Agent spawns at a random cell each episode; combined with randomized food, the agent-to-food distance is fresh every episode, so search efficiency can't be a fixed spawn-to-food artifact. |

### 1.3 The internal-state dial (induce the behavior)

Foraging is driven by depletion. Two starting conditions, run as a within-probe contrast:

- **Sated start** (`start_satiation: 100`, `start_nutrition: 100`): baseline. A well-regulated agent
  should explore/forage *little* early (already full) then begin searching as metabolism depletes it.
- **Hungry start** (`start_satiation: 40`, `start_nutrition: 40`): induced motivation. The agent
  begins depleted, so a genuine forager should set off to search for food promptly.

The contrast between these two is the heart of the probe: search-and-forage that responds to need is
the signal; identical search effort regardless of the starting dial is reflexive wandering.

### 1.4 Candidate mean-level measures

The sparse-food world splits foraging into two phases — a **search** phase (food not yet perceived,
the agent must cover ground to find it) and an **approach** phase (food within `cue_radius`, the
agent must move toward it). The measures cover both.

| Measure | Definition (per episode, then mean over 200 eval episodes) | Phase / reads on |
|---|---|---|
| **F1 — time-to-first-eat** | Steps until the first `ate_food` event. With sparse food this is a **search-efficiency** read (time to locate *and* reach the food). Lower = more efficient foraging. | Search; both starts; expect hungry ≪ sated. |
| **F2 — eat rate** | Fraction of steps with `ate_food` over the episode. With respawn-on-consumption this is sustained foraging throughput (find → eat → find again). | Both starts. |
| **F3 — satiation recovery slope** | Linear slope of `drive_hunger` (or satiation) over the first 100 steps from the hungry start. Positive = restoring. | Hungry start only. |
| **F4 — directed-approach fraction** | Of steps where food *is* within `cue_radius` but not yet eaten, the fraction where the agent's next move *reduces* distance to the food. Separates "wanders into food" from "goes to food once it can see it". | Approach; both starts. |
| **F5 — exploration coverage rate** | Unique cells visited per step *before the first food contact* (equivalently, fraction of the arena covered while searching). A systematic searcher covers ground; a freezer/tight-looper does not. | Search; both starts; expect hungry > sated early. |

F1, F4 and F5 are the headline triad: **F5** says "did it actually search the world", **F1** says "did
that search find food efficiently", **F4** says "once it could see food, did it go *toward* it on
purpose". Together they distinguish a real explore-then-forage policy from both freezing and random
wandering.

**Reading the size sweep:** plot F1 (search efficiency) and F5 (coverage) against grid size. The diagnostic claim is comparative — *"the agent forages successfully at 5×5 and 10×10 but its search collapses at 15×15"* — which pins down the agent's effective search horizon, not just a pass/fail at one size.

### 1.5 Anti-confound clause (foraging)

Foraging has no threat, so the spatial-encounter artifact does not apply directly — but a sibling
artifact does: **F1/F2 could be a pure spawn-geometry effect** (agent happens to spawn near the food →
eats fast, with no goal-directed search). Three design moves rule this out:

1. **Single food at a randomized location + `random_start_pos: true`** independently randomizes the
   agent-to-food distance every episode, so a fast time-to-eat averaged over 200 seeds reflects search
   skill, not one lucky spawn.
2. **F4 is geometry-immune.** A wanderer's next-move-reduces-distance fraction is ~chance; a directed
   approacher's is well above chance — independent of absolute positions.
3. **F5 cannot be faked by camping.** A freezer covers ≈ 0 new cells and never finds the sparse food
   (long F1); only an agent that genuinely explores scores high F5 *and* low F1.
4. **The hiding predator must not become an avoidance confound.** It is stationary, so it creates no
   pursuit dynamics, but its contact damage *could* still distort F5 if the agent fixates on routing
   around hazard cells instead of foraging. Guard: keep it sub-lethal (`damage: [15, 45]`, never the
   chasing predator's lethal range) and at low count, and let the §1.6 trajectory cross-check confirm
   the agent is *foraging*, not hazard-fixating. If avoidance dominates, drop the hiding predator to
   `count: 0` for that run — it is a distribution-matching convenience, not a load-bearing factor.

> **What would make F-measures a geometry artifact:** if the food spawned near the agent's spawn cell,
> F1/F2 would look high regardless of policy. Ruled out by *independent* randomization of food and
> agent positions, by F4 (a per-step directional measure), and by F5 (which rewards covering ground,
> not sitting near food).

### 1.6 Trajectory cross-check (mandatory before trusting any F-mean)

Run `trajectory-story` on ~5 episodes per start condition and confirm:

- In the **hungry start**, the step dump shows the agent **systematically exploring** the arena (F5 —
  covering new ground, not freezing or looping in a corner), then turning to move *monotonically
  toward* the food once it enters `cue_radius` (F4) — both means must match what the steps show.
- In the **sated start**, the agent should search *less* early; if F1/F5 are identical across starts,
  the dial isn't driving behavior and the foraging read is reflexive, not regulated.
- The eat events in the dump line up with satiation upticks (F3 slope is real, not a logging artifact).
- The agent treats the **hiding predator** as a minor hazard, not the main event — it steps around the
  trap cell when convenient but does not spend the episode hazard-fixating instead of foraging (if it
  does, the hiding predator is confounding the read — see §1.5 bullet 4).

Only if the trajectory tells the same story as F1/F4/F5 is the foraging verdict trusted.

---

## 2. Avoidance — keep distance from danger, not from a harmless look-alike

### 2.1 The behavior in one sentence

Does the agent keep more distance from / disengage faster from an entity that **hurts it** than
from a behaviorally identical entity that **does not** — and is that gap a real class response
rather than a geometry artifact?

### 2.2 The isolating world (geometry-controlled by construction)

This is the family the discrimination search burned us on, so the world is built *anti-confound-first*.
One damaging predator and one harmless rabbit, **byte-identical in every observable and chase field**
except damage (the cell-08 contrast), placed in a **mirror-symmetric arena** so neither class has a
spatial-encounter advantage:

| Knob | Value | Why |
|---|---|---|
| Grid | 11×11 (odd, so a true center column exists for mirror symmetry) | Mirror counterbalancing needs a symmetric axis. |
| Predator | count 1, `behaviour: hunt`, damaging `[5,120]` | The threat. |
| Rabbit | count 1, `behaviour: hunt`, harmless `[0,0]`, otherwise identical | The matched control — same smell, speed, detection, stamina. |
| **Spawn symmetry** | predator and rabbit spawn in **mirror-image halves**; run two configs (predator-left/rabbit-right and predator-right/rabbit-left), average them | Cancels any side bias and equalizes encounter geometry — the exact control the search's "arena_PL/PR" pair used. |
| Food | symmetric, both halves equally provisioned | Neither class sits closer to the resource the agent must visit. |
| `random_start_pos` | true | Agent's start is symmetric in expectation. |

The hunt behaviour (vs. static) is deliberate: it gives both animals the *same* mobility so the
predator cannot win the gap merely by roaming more (the precise mechanism that produced the artifact).

### 2.3 The internal-state dial

- **Mild hunger** (`start_satiation: 60`): the agent must leave any safe spot to forage, so it is
  *forced into encounters* with both classes rather than camping a corner (camping was the Cell A1
  degenerate that made avoidance untestable). This guarantees non-zero encounter denominators for
  both classes — the prerequisite for a valid contrast.

### 2.4 Candidate mean-level measures (all class-contrasted, predator − rabbit)

| Measure | Definition | H₁ direction |
|---|---|---|
| **A1 — Δ closest-approach distance** | mean over episodes of (min distance ever reached to rabbit) − (min distance to predator). Positive = stays farther from predator. | > 0 |
| **A2 — Δ disengage latency** | after an entity first enters `cue_radius`, steps until the agent's distance to it exceeds `cue_radius` again. predator − rabbit. Negative = disengages predator faster. | < 0 |
| **A3 — M2 bush-dive gap** | `bush_dive_rate_predator − bush_dive_rate_rabbit` (reuse existing M2). | > 0 |
| **A4 — class-blind control prediction** | the SAME measures computed for an entity-label-shuffled rollout (predator/rabbit labels swapped post-hoc). Must collapse to ≈ 0 if the gap is class-driven. | ≈ 0 |

### 2.5 Anti-confound clause (avoidance) — the load-bearing section

> **What would make A1–A3 a geometry artifact:** if the predator encountered the agent in threatening
> contexts more often than the rabbit *for spatial reasons* (it roams more, it sits nearer the food,
> it spawns nearer the agent), then ANY agent — even a class-blind one — shows a predator-avoidance
> gap. This is exactly what happened in the discrimination search (+0.31 gap reproduced by a known
> class-blind agent).

Four design moves rule it out, and a gap is only believed if **all four** hold:

1. **Mirror-symmetric, counterbalanced spawn** (predator-left + predator-right averaged): encounter
   geometry is equalized by construction. A residual gap cannot be a side bias.
2. **Identical mobility** (both `hunt`, identical chase fields): the predator cannot win by roaming
   more than the rabbit.
3. **Encounter-frequency reporting (mandatory diagnostic).** Per episode, log how often each class is
   within `cue_radius` (`A4-denom`). The gap is only interpretable if predator-near-fraction ≈
   rabbit-near-fraction (within, say, 20%). If the predator is near more often, the world is *not*
   geometry-controlled and the measure is rejected, not reported.
4. **Class-blind control prediction (A4).** Re-evaluate a *known class-blind* agent (the cell-08 final
   checkpoint) in this same world. If it shows the gap, the world is still leaking geometry. The world
   is only certified once a class-blind agent reads ≈ 0 on A1–A3 here.

This is the calibrate-before-trust gate the search established, now baked into the world's design
rather than discovered after the fact.

### 2.6 Trajectory cross-check (mandatory)

Per the search's own correction: confirm via step dumps that (a) both classes actually approach the
agent at comparable rates (the encounter-symmetry the means assume), and (b) the flee-decomposition is
class-specific — the agent flees the close-range *predator* more than the close-range *rabbit* at
matched distance. If at distance-1 the agent flees both equally (the class-blind signature), the A1–A3
means are a residual to distrust no matter their sign.

---

## 3. Recovery — restore a depleted internal state after a disturbance

### 3.1 The behavior in one sentence

After we drive an internal state away from its setpoint (injure it, or starve it), does the agent take
*active corrective action* to restore that state — feed, rest, retreat to cover — rather than passively
drifting back?

### 3.2 The isolating world

Recovery needs a **disturbance, a remedy, and no competing pressure**. The world has food (the remedy
for hunger), bushes/safe cover and the rest action (the remedy for injury), and **no live threat** —
so the only thing to do after a disturbance is recover. Two sub-worlds, one per internal axis:

| Sub-world | Disturbance dial | Remedy present | Threat |
|---|---|---|---|
| **R-hunger** | `start_satiation: 20`, `start_nutrition: 20` (deep depletion) | 8 food sources (symmetric) | none (predator + rabbit count 0) |
| **R-injury** | `start_injury: 60` (well above resting, below the 100 death line) | rest action + bushes (cover); food still present for metabolism | none |

Grid 10×10, `random_start_pos: true`. R-injury uses the existing recovery machinery (`recovery_base_rate`,
`recovery_accel_rate`, `rest_action_enabled`) unchanged.

### 3.3 Candidate mean-level measures

| Measure | Definition | Reads on |
|---|---|---|
| **C1 — recovery half-time** | steps for the disturbed state to return halfway to setpoint. Lower = faster active recovery. | both sub-worlds |
| **C2 — corrective-action fraction** | of the first K steps after spawn, the fraction spent on the *remedy* action (eating for R-hunger; resting/in-bush for R-injury) vs. unrelated moves. | both |
| **C3 — overshoot / regulation** | does the agent stop the corrective action near setpoint (good regulation) or overshoot (e.g. overeat)? Measured as state value at the step corrective action ceases. | R-hunger |

C2 is the headline: it separates *active* recovery (the agent chooses the remedy) from *passive* drift
(injury decays on its own via `injury_smoothing` / metabolism refills happen to occur).

### 3.4 Anti-confound clause (recovery)

> **What would make C-measures an artifact:** the internal states recover *on their own* — injury
> smooths out over `injury_smoothing_duration`, and satiation is a deterministic function of nutrition.
> So a passive agent that never takes the remedy action could still show C1 (half-time) improving,
> faking "recovery".

Ruled out by:

1. **C2 is the active-choice measure**, not C1. C2 counts remedy actions; passive decay contributes
   nothing to it. The recovery claim rests on C2 above a passive baseline.
2. **Passive baseline (control prediction).** Compute C1/C2 for a *do-nothing* policy (scripted: always
   `stay`) in the same world. The trained agent's C1 must beat the do-nothing C1, and its C2 must be
   well above the do-nothing C2 (≈ 0), or "recovery" is just the world's built-in decay.
3. **Random spawn** so corrective-action fraction can't be a fixed spawn-on-food artifact.

### 3.5 Trajectory cross-check (mandatory)

Step dumps must show the disturbed state *and* the action stream together: in R-hunger, eat events
should cluster early and stop near setpoint (C3); in R-injury, rest/bush-occupancy should spike right
after spawn while injury is high and relax as it normalizes. If injury falls but the agent never rests
or hides, C1 is pure passive decay and the recovery verdict is "passive only".

---

## 4. Managing conflicting needs — eat the nearby food vs. flee to the far cover

### 4.1 The behavior in one sentence

When the agent is **both hungry and injured** and a predator is closing in, does it grab the **nearby
food** (relieve hunger, but stay exposed) or commit to the **distant bush** (escape the threat, but
abandon the food)? Food and cover are placed **far apart**, so the agent cannot do both — the choice
is a single, concrete, observable decision.

### 4.2 The isolating world

The conflict is made **spatial**: the two ways to relieve the two drives sit at opposite ends of the
world, with the agent between/near one of them and a predator approaching.

| Knob | Value | Why |
|---|---|---|
| Grid | **7×7** (small — sized for isolation, not scale) | Smallest grid where food and bush are clearly separate directions: opposite corners are ~8 cells apart (Manhattan), well beyond the agent's **cue radius of 3**, so the two options are mutually exclusive — committing to one abandons the other. Smaller (5×5) puts both inside reach and the choice isn't forced. |
| Food | **one** source, placed **next to the agent's start** | Hunger can be relieved immediately *if* the agent chooses to stay and eat. |
| Bush | **one** bush (`hides_agent: true`), distance from the food **swept: near / middle / far** (≈3 / 5 / 8 cells) on the same 7×7 | The only refuge from the predator. Sweeping its distance turns the **flee-cost** into the independent variable: cheap escape (near) vs. expensive escape (far). |
| Predator | count 1, `hunt`, damaging, spawned so it **approaches** the agent/food area | The closing threat that forces the timing of the decision. |
| Rabbit | count 0 | Isolate eat-vs-flee; no harmless distractor. |
| `start_pos` | **fixed, next to the food** (not random) | We want the agent to begin within reach of food so the decision (stay+eat vs. run to cover) is clean and comparable across seeds. |

The **swept independent variable is the bush distance** (near/middle/far) — three configs on the same
grid. Everything else (start, food, predator approach, internal state) is held fixed, so any change in
the choice is attributable to how far safety is.

### 4.3 The internal-state dial — both drives switched ON

A single, deliberately conflicted starting state (not a sweep): **low nutrition + high injury**.

- **Low nutrition** (`start_nutrition` low → low satiation): strong drive to **eat** the nearby food.
- **High injury** (`start_injury` high, well below the death line): the agent is already hurt, so a
  further predator hit is near-lethal — a strong drive to **escape to cover**.

Both drives are active at once and point at **different locations**, which is the conflict. (For
interpretability, run a small set of **control conditions** alongside — e.g. *no predator* and
*no injury / sated* — to confirm the agent goes to food when nothing opposes it; the bush-choice is
only meaningful as a *change* from that baseline.)

### 4.4 Candidate mean-level measures

| Measure | Definition (per episode → fraction/mean over eval episodes) | Reads on |
|---|---|---|
| **K1 — choice outcome** | Which target the agent commits to first: **eats the food** vs. **reaches the bush** vs. **neither (dies/wanders)**. The headline categorical measure. | the decision itself |
| **K2 — time-to-commit** | Steps until the agent is unambiguously heading to one target (distance to it decreasing monotonically while the other increases). | decisiveness vs. dithering |
| **K3 — eat-then-flee vs. flee-only** | Does it grab a bite *then* run to cover, or abandon food entirely? | how it sequences the two needs |
| **K4 — survival outcome** | Did the chosen action keep it alive (reached cover before a lethal hit / ate without being caught)? | the cost of the choice |

K1 is the headline: under hunger+injury+approaching-predator, *what does it actually choose*. K2–K4
describe how cleanly and at what cost.

**K5 — flee-cost flip point (the across-variant headline).** Plot K1 (flee-to-bush rate) against the near/middle/far bush distance. A genuine trade-off **declines** as the bush gets farther (escape gets too costly → the agent stays and gambles on eating); the *distance at which the choice flips* is a single interpretable number for how the agent prices safety against hunger+injury. A flat curve = the choice ignores flee-cost (reflexive).

### 4.5 Anti-confound clause (conflict)

> **What would make the choice an artifact:** if the predator's approach path physically **blocked**
> the route to the food (or to the bush), the "decision" would just be which target was reachable —
> geometry, not a weighed trade-off.

Ruled out by:

1. **Both targets must stay genuinely reachable** at decision time — position the predator so it
   threatens the food area without walling off either the food or the bush. Verify in the trajectory
   that an unblocked path to *both* existed when the agent chose.
2. **Control conditions** (§4.3): if the agent goes to food under *no predator* and to the bush only
   when the predator + injury are present, the choice is threat/state-driven, not geometry.
3. **Far, symmetric separation**: food and bush equidistant-ish from the start in opposite directions,
   so neither is "on the way" — the agent must actively pick a direction.

### 4.6 Trajectory cross-check (mandatory)

Read the step dump: the agent should **commit to one target** (monotonic approach), not oscillate
between them. Confirm K1 against what the steps show — e.g. at high injury + closing predator it turns
*away* from the nearby food and runs the long way to the bush (survival overrides hunger), whereas in
the *no-predator* control it walks straight to the food. If the episodes look the same with and without
the threat, the conflict read is "state-insensitive" and K1 is not measuring a trade-off.

---

## 5. Why these four worlds avoid the search's trap

| Family | The artifact that would fool it | The design feature that rules it out |
|---|---|---|
| Foraging | spawn-near-food luck | random spawn + F4 (per-step directional, position-free) |
| Avoidance | predator encounters agent more (geometry) | mirror counterbalance + identical mobility + encounter-frequency check + class-blind control |
| Recovery | states decay on their own | C2 active-choice measure + do-nothing passive baseline |
| Conflict | food region is "on the way" | K3 is a *slope across need*, which constant geometry can't produce |

The common thread: **the headline measure in every world is one that a fixed geometry cannot
manufacture** — a per-step directional fraction (F4), a counterbalanced + control-gated class contrast
(A-family), an active-action count vs. a passive baseline (C2), or a slope across a deliberately swept
internal dial (K3). This is the structural answer to
[[20260616_0142_discrimination_is_spatial_encounter_artifact]].

---

## 6. Open design questions (for the user to decide)

1. **Checkpoint set.** Which trained agents do we probe first? The discrimination search left us with
   *no confirmed genuine discriminator* — so the avoidance probe (§2) may, on the agents we have,
   correctly read ≈ 0. Is the v1 goal to (a) demonstrate the probes work on agents whose behavior we
   already understand (cell-08 class-blind, Cell A1 camping), or (b) wait for a genuinely
   class-discriminating agent before building the avoidance probe?
2. **Avoidance probe priority.** Given (1), should §2 (avoidance) be deferred until a learnable-distal-cue
   agent exists, and v1 ship Foraging + Recovery + Conflict first (the three that do not require a
   discriminator to be meaningful)?
3. **Passive/do-nothing baseline mechanism.** §3 and §2 both call for scripted control policies
   (do-nothing; class-label-shuffle). Does `scripts/eval/eval_rollout.py` support a scripted policy, or is
   that a `senior-developer` feature request before these probes can run?
4. **Grid-size variants.** I propose one shrink variant per family for size-invariance. Is that worth
   the extra configs in v1, or defer size-invariance to a later robustness study?
5. **New measures F1/F3/F4, A2, C1–C3, K1/K3/K4.** Several proposed measures are not in the existing
   M1/M2/M5/M7 toolkit (which covers A3=M2 and K2=M5). These need either offline computation from the
   eval-rollout dumps or new online accumulators. Should I file a `## Metrics Requested` follow-up to
   route the new ones through `senior-developer`, or restrict v1 to measures the current toolkit
   already emits (M2, M5, distance, survival)?

---

## 7. Config manifest (planned — NOT yet written)

Status: **planned.** Each row is one config file a follow-up step would create under
`configs/experiment/behavior_measures/`, expressed as deltas from an existing testbed config. All
inherit the cell-08 sensor/body/noise/`behavior_measures` blocks unchanged unless noted.

| # | File (planned) | Base config | Key deltas |
|---|---|---|---|
| 1 | `forage_sated.yaml` | `testbed_forage_only.yaml` | grid → 10×10 (same as training); add one static predator-class animal (`class: predator`, `behaviour: static`, `count: 1`, `damage: [15, 45]`, `nociception_intensity: 0.9`) — the hiding predator in the unified animal system; chasing (`behaviour: hunt`) predator + rabbit `count: 0`; food `count: 1` (single source), `random_food_pos: true` + respawn-on-consumption; `random_start_pos: true`; sated start (`body.start_satiation:100`, `body.start_nutrition:100`) — this is the anchor. |
| 2 | `forage_hungry.yaml` | row 1 | `body.start_satiation: 40`, `body.start_nutrition: 40`. |
| 3a | `forage_hungry_5x5.yaml` | row 2 | `environment.height: 5`, `environment.width: 5` (**easy** variant) — food almost always within `cue_radius`; **expect success**. |
| 3b | `forage_hungry_15x15.yaml` | row 2 | `environment.height: 15`, `environment.width: 15` (**hard** variant) — single food rarely encountered; the size at which search **fails** for a fragile forager. Rows 2 (10×10, same as training) + 3a (5×5, easy) + 3b (15×15, hard) form the size sweep (DOF #1 size-invariance test). |
| 4 | `avoid_mirror_PL.yaml` | `testbed_arena_PL.yaml` | grid → 11×11; predator+rabbit `behaviour: hunt` (was static); predator spawn/patrol = left half, rabbit = right half; food symmetric both halves; `body.start_satiation: 60`. |
| 5 | `avoid_mirror_PR.yaml` | `testbed_arena_PR.yaml` | mirror of row 4: predator = right half, rabbit = left half. (PL+PR averaged.) |
| 6 | `recover_hunger.yaml` | `testbed_forage_only.yaml` | `body.start_satiation: 20`, `body.start_nutrition: 20`; predator+rabbit count 0 (already). |
| 7 | `recover_injury.yaml` | `testbed_forage_only.yaml` | `body.start_injury: 60`, `body.random_start_injury: false`; predator+rabbit count 0; food kept for metabolism. |
| 8a | `conflict_bush_near.yaml` | `08-singlePredRabbit_disengage.yaml` | grid → 7×7; rabbit count 0; **one** food next to a **fixed** `start_pos` (corner); **one** bush (`hides_agent: true`) **~3 cells** away; **one** `hunt` predator on a flank (threatens without blocking either path); low nutrition + high injury start (`body.start_nutrition` low, `body.start_injury` high, `random_start_*: false`). |
| 8b | `conflict_bush_mid.yaml` | row 8a | bush moved to **~5 cells** from the food. |
| 8c | `conflict_bush_far.yaml` | row 8a | bush moved to the **opposite corner (~8 cells)**. |
| 9 | `conflict_ctrl_nopred.yaml` | row 8c | control: predator `count: 0` (expect → food regardless of bush distance). |
| 10 | `conflict_ctrl_sated.yaml` | row 8c | control: sated + uninjured start (expect → food / no urgency). |

**Pre-flight requirement:** before any of these launch, the generated configs go through
`env-config-auditor` (obs↔noise sync, `hides_agent` present for M2, `eval_seeds` length = 200, no
tag collisions, mirror symmetry for rows 4–5). No probe runs until the auditor passes.

---

## 8. Links

- Governing perspective: [[experiment_environment_design_perspective]] (this doc is its first concrete
  instalment).
- The lesson every threat design defends against:
  [[20260616_0142_discrimination_is_spatial_encounter_artifact]].
- Prior testbed search (method + verdict): [[testbed_solo_validation_results]].
- Measure vocabulary reused (M2 bush-dive, M5 eat-under-threat, M7 motifs):
  [[behavior_measure_toolkit_v1_design]].
- Base configs the manifest forks: `configs/experiment/hypervigilance/{testbed_forage_only,
  testbed_arena_PL,testbed_arena_PR,08-singlePredRabbit_disengage}.yaml`.
