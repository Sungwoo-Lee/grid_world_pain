---
title: "SameProp Round 4 — chasing rabbit: does the agent flee a harmless animal that actively chases it?"
topic: hypervigilance
status: active
created: 2026-05-29
last_updated: 2026-06-01
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

> **PREVIEW — mid-training snapshot at ~74% (~7.44M / 10M episodes). The run is
> still training on node 110. Nothing here is a closing verdict; the numbers can
> still move. Treat the direction and the class contrasts as the signal, not the
> absolute levels.**

### 6.0 Headline (plain English, read this first)

We made the two harmless rabbits actively **chase** the agent across the whole map,
to test whether the agent's avoidance of threats is driven by the *chasing motion*
or by *actual danger*. The single question was: **does the agent flee a harmless
animal that chases it?**

**At 74% of training, the answer is no -- the agent does not flee the chasing
rabbits.** It treats them very differently from the genuinely dangerous predator,
in the direction of *tolerating* them:

- **It keeps the predator at arm's length but lets the rabbits come right up to it.**
  Averaged over the most recent third of training, the agent stays about **4.8
  cells** from the predator but only about **3.1 cells** from each rabbit -- and the
  rabbit distance has been *shrinking* across training (it was ~3.6-3.7 early on),
  i.e. the agent is getting *more* comfortable with the chasers, not fleeing them.
- **It dives for cover far more often for the predator than for a rabbit.** The
  "dive into a bush when a threat closes in" rate is about **0.40 for the predator**
  versus about **0.25-0.27 for each rabbit**, and the rabbit number is *not* climbing
  toward the predator number despite the rabbits now chasing exactly like the predator.
- **It absorbs an enormous number of harmless rabbit contacts.** The agent now
  collides with the rabbits roughly **62-75 times per episode** (and climbing),
  versus only ~3.5 contacts with the predator. It has clearly learned that letting a
  rabbit touch it costs nothing, while a predator touch hurts.

So the agent is reading *something other than "is it chasing me"* to decide what to
avoid -- most consistent with a **learned per-animal danger association** (the
predator has hurt it; the rabbits never have), not a generic "chasing = threat"
heuristic.

**What this means for the original measure-validity worry.** The concern that
launched this round was that our main avoidance measure (the bush-dive rate, "M2")
might be *blind to flight* -- if the agent fled rabbits by running rather than diving
into cover, M2 would under-count rabbit-avoidance and we'd wrongly conclude the agent
ignores rabbits. **This preview largely retires that worry for this world.** We
checked the flight channel directly -- the distance the agent keeps from each animal --
and it tells the *same* story as the bush-dive rate: the agent keeps the predator
farther and the rabbits closer, and is moving *closer* to the rabbits over time.
There is no hidden flight-from-rabbits that the bush-dive measure was missing. Both
channels agree the agent does not avoid the harmless chasers.

**Caveats, stated honestly.** (1) This is one seed (random seed 42), mid-training,
not converged. (2) These are **training-time online metrics** -- averaged over the
exploring, stochastic policy as it learns -- and so are **not directly comparable in
absolute level** to the seed-locked, deterministic eval-rollout baselines quoted for
the passive-rabbit world (those replay 200 fixed seeds with a greedy policy). The
honest comparison is *direction and class-contrast*, not number-vs-number. (3) A
deterministic eval-rollout on a final checkpoint is still needed to lock this in.

### 6.1 What was measured, and against what

| | This run (R4 chasing rabbit) | Passive-rabbit baseline (R2.6 Cell C) |
|---|---|---|
| World | 3 moving threats: 1 damaging predator + 2 **harmless chasing** rabbits | predator + 2 **wandering** (passive) rabbits |
| Rabbit behaviour | `hunt`, detect-anywhere, persistent | `wander` |
| Metric source | **online training-time** averages (stochastic policy, still learning) | **deterministic eval-rollout** (200 fixed seeds, greedy policy) |
| Comparability | direction / class-contrast only | -- |

The baseline class gap in the passive world (deterministic eval): the agent dived
for cover ~0.89 of the time for the predator vs ~0.51 for a rabbit (a +0.37 gap),
and kept the predator farther than the rabbit. The R4 question was whether making the
rabbits *chase* would close that gap. It has not.

### 6.2 Metric keys actually logged (this run)

Per-class and **per-tag** behaviour-measure keys are all present (tags: predator
`full`, rabbit `TL`, rabbit `BR`):

- **Distance / flight channel** -- `Episode/MeanDistPredator_full`,
  `Episode/MeanDistRabbit_TL`, `Episode/MeanDistRabbit_BR`
  (plus class-level `MeanDistPredator`, `MeanDistRabbit`, and `MeanDistFood`,
  `MeanDistHidingPredator`).
- **M2 bush-dive channel** -- `Episode/BushDiveRate_{predator_full,rabbit_TL,rabbit_BR}`
  with denominators `Episode/BushDiveDenominator_{predator,rabbit}` (the
  encounter-count the rate is normalised by).
- **M5 eat-under-threat** -- `Episode/EatUnderThreatRate_{predator_full,rabbit_TL,rabbit_BR}`
  and `...Ratio...`.
- **M1 interrupted-feeding** -- `Episode/InterruptedFeedingRate_{predator_full,rabbit_TL,rabbit_BR}`.
- **Contact / damage accounting** -- `Episode/PredatorHits`, `Episode/RabbitHits`,
  `Episode/DamagePredator`, `Episode/DamageObstacle`, `Episode/TotalDamage`,
  `Episode/Term_{Injury,MaxSteps,Starvation}`.
- **Survival** -- `Episode/Steps`.

### 6.3 Temporal evolution (training thirds, online metrics)

The 73 logged history records were split into three equal-count windows (early / mid
/ late by training order). Values are window means.

| Metric | early | mid | late | direction |
|---|---|---|---|---|
| `Episode/Steps` (survival) | 105.1 | 154.0 | **229.4** | rising strongly -- env not crippling |
| `MeanDistPredator_full` (cells) | 5.13 | 5.17 | **4.84** | ~flat / slight down |
| `MeanDistRabbit_TL` (cells) | 3.72 | 3.42 | **3.08** | down -- agent moving **closer** to chaser |
| `MeanDistRabbit_BR` (cells) | 3.55 | 3.39 | **3.08** | down -- agent moving **closer** to chaser |
| `BushDiveRate_predator_full` | 0.332 | 0.366 | **0.403** | up |
| `BushDiveRate_rabbit_TL` | 0.235 | 0.218 | **0.271** | ~flat -- **not** chasing predator level |
| `BushDiveRate_rabbit_BR` | 0.180 | 0.210 | **0.266** | mild up but stays below predator |
| `EatUnderThreatRate_predator_full` | 0.029 | 0.053 | 0.083 | up (longer episodes -> more threat-overlap feeding) |
| `EatUnderThreatRate_rabbit_TL` | 0.050 | 0.090 | 0.140 | up -- agent eats *more* freely near rabbits |
| `EatUnderThreatRate_rabbit_BR` | 0.049 | 0.089 | 0.140 | up -- same |
| `InterruptedFeedingRate_predator_full` | 0.910 | 0.900 | 0.875 | ~flat high |
| `InterruptedFeedingRate_rabbit_TL` | 0.946 | 0.927 | 0.923 | ~flat high* |
| `PredatorHits` (contacts/ep) | 1.65 | 2.30 | 3.52 | up (more steps alive) |
| `RabbitHits` (contacts/ep) | 21.1 | 38.5 | **62.4** | up up -- agent **tolerates** harmless contact |
| `DamagePredator` | 49.6 | 68.9 | 105.6 | up -- all damage is predator/obstacle |
| `Term_MaxSteps` (frac reaching 500) | 0.000 | 0.014 | 0.137 | up -- agent starting to survive full episodes |

\* The interrupted-feeding rate is high for **all** animals including rabbits, but
this measure fires whenever *any* qualifying animal is within the cue radius during a
feeding attempt -- with three constantly-roaming threats on a 10x10 grid the cue
radius is almost always occupied, so a uniformly-high interrupted-feeding rate is the
expected geometry artifact, not evidence of rabbit-specific caution. The discriminating
channels here are **distance** and **bush-dive**, both of which separate predator from
rabbit cleanly.

### 6.4 The two avoidance channels agree (the measure-validity result)

The original worry was that the bush-dive measure (M2) only counts dives into cover
and would miss *flight* (running away while keeping distance). With chasing rabbits,
flight was the predicted way the agent might avoid a rabbit without diving. We
therefore read both channels:

- **Bush-dive channel:** predator ~0.40, rabbits ~0.25-0.27, gap persists, rabbit
  rate not climbing to predator level.
- **Flight / distance channel:** predator kept ~4.8 cells away, rabbits allowed to
  ~3.1 cells and *closing further* over training.

Both channels point the same way: **no rabbit-avoidance, strong predator-avoidance.**
There is no flight-from-rabbits hiding behind a flat bush-dive number. For this world,
the bush-dive measure was *not* misleading -- it agreed with the independent flight
read. This is the main measure-validity takeaway.

### 6.5 Encounter-geometry asymmetry -- now resolved

A prior analysis flagged that passive rabbits rarely *came to* the agent, so the
rabbit denominators (encounter counts) were tiny relative to the predator's, making
rabbit avoidance rates noisy and hard to compare. With the rabbits now chasing, the
denominators are comparable -- latest snapshot: bush-dive denominator ~**5.0 for the
predator** vs ~**5.3 for the rabbits** (rabbits now slightly *exceed* the predator
in encounter count). The asymmetry we worried about is effectively gone, which makes
the class contrast above more trustworthy than the passive-world comparison was.

### 6.6 Sanity check -- rabbits never deal damage

Confirmed. Despite 62-75 rabbit contacts per episode, **all** logged damage is
attributable to the predator (`DamagePredator` ~106, rising with episode length) and
static obstacles (`DamageObstacle` ~25); `RabbitHits` carries no damage. The neutral
class kept `is_damaging = False` as designed -- the chase was the only manipulation.

### 6.7 Provisional conclusion (mid-training)

Against the two informal predictions in section 2:

- **"Avoidance rises toward predator-level because of chasing motion"** -- **not
  supported** at 74%. Chasing the agent did not pull rabbit avoidance up; the
  agent reads chasing-motion as *not* a danger cue on its own.
- **"Agent learns the chaser is harmless and approaches/ignores it"** -- **supported**
  so far. Rabbit distance is shrinking, rabbit contacts are soaring with zero damage,
  and bush-dives for rabbits stay well below predator level.

The most parsimonious reading: in this matched-smell world the agent discriminates
threats by a **learned, per-animal danger association** (what has actually hurt it),
**not** by the generic "is-this-thing-chasing-me" motion cue. One seed, mid-training --
a deterministic eval-rollout on a converged checkpoint is the confirmation step before
this is stated without hedging.

### 6.8 Follow-ups / TODO

- **Confirm with a deterministic eval-rollout** on a late/final checkpoint (the
  config already carries the 200-seed eval list and `eval_policy_mode: deterministic`)
  so the levels are directly comparable to the R2.6 baseline rather than
  direction-only.
- **Re-read at 100%** to confirm the rabbit-distance downtrend and bush-dive gap hold
  to convergence.
- Consider a **multi-seed** repeat before any published claim -- this is a single-seed
  pilot by design.

_Analysis is post-fill of a lightly-pre-registered exploratory design; section 2
predictions were informal ("watching, not gating"), so these conclusions are
observational, not a thresholded confirm/refute._

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


---

## No-predator transfer eval

**Plain-language purpose.** We trained an agent in a world that contained one
genuinely dangerous patrolling predator plus two harmless rabbits that chase it but
never hurt it. This follow-up asks a clean transfer question: if we now drop the agent
into a world where **nothing can hurt it** — every damaging entity removed, only the
two harmless chasing rabbits left — does it keep behaving the way it learned to? If the
agent still flees, hides in bushes, or keeps its distance from the rabbits even though
they are demonstrably harmless and no real threat exists, that leftover caution is a
**learned policy habit**, not a rational response to present danger. If instead it
relaxes — forages freely, ignores the rabbits — then its caution during training was
threat-contingent, not a fixed defensive disposition. This is an **evaluation only** (a
rollout of the already-trained checkpoint); no new training happens.

**What was removed (vs the chasing-rabbit training world).** Two deletions, nothing
else: (1) the single patrolling predator — the only entity that could actually deal
damage; and (2) the four "bush-ambusher" damage sources (`hiding_predator` resources)
that lurked in the corners. The two chasing rabbits stay exactly as trained — same
full-grid pursuit, same matched smell, still harmless. The food layout, the visible
bush / rock / tree obstacles (including the bushes the agent can dive into to hide), the
sensors, the body model, the noise profile, and the evaluation protocol are all kept
byte-identical so the only thing that changes between train and eval is **the presence
of threat**.

**Why the trained checkpoint still loads (observation-dimension parity).** A network
trained on one config can only be evaluated on another if the observation vector has the
**same shape**. Removing entities is safe here because the agent's visual sense encodes
entities by **class**, not by individual — there is one channel for "a predator is
visible", one for "a neutral animal is visible", and so on. Deleting the predator just
leaves its class channel reading empty; it does not delete a channel. The observation
breakdown is therefore identical in both worlds: 27 numbers total (1 satiation, 1 felt
internal pain, 1 external pain signal, 5 smell, 5 collision, 6 proprioception, 8 vision).
Verified by loading both configs through the environment loader — see the parity table
below.

### Methods / manifest detail

- **Eval config:** [`configs/experiment/hypervigilance/05-noPredator_chasingRabbit_eval.yaml`](../../../../configs/experiment/hypervigilance/05-noPredator_chasingRabbit_eval.yaml)
- **Trained on (checkpoint source):** [`configs/experiment/hypervigilance/04-sameProp_R4_chasingRabbit.yaml`](../../../../configs/experiment/hypervigilance/04-sameProp_R4_chasingRabbit.yaml)
- **Surgical diff vs 04:** removed the 4 `hiding_predator` resources from `resources:`,
  and removed the `class: predator` (tag `full`) entity from `entities:`. Header comment
  rewritten to describe the eval. Everything from `body:` onward is byte-identical (diff
  confirmed empty).

**Loader validation (config 04 vs 05):**

| Check | Config 04 (train) | Config 05 (no-predator eval) | Verdict |
|---|---|---|---|
| Obs breakdown | `{Satiation:1, InteroNoci:1, ExteroNoci:1, Olfaction:5, Collision:5, Proprio:6, Visual:8}` | identical | parity PASS |
| Obs dim total | 27 | 27 | PASS (checkpoint loads) |
| `predator_indices` | `(0,)` | `()` | zero-predator PASS |
| `neutral_indices` | `(1, 2)` | `(0, 1)` | the two rabbits PASS |
| `animal_is_damaging` | `[True, False, False]` | `[False, False]` | nothing damaging PASS |
| `behavior_measures.enabled` | true | true | PASS |
| obstacle with `hides_agent:true` & count>0 | 2 | 2 | M2 bush-dive support PASS |

**Zero-predator edge case.** With no predator-class entity the loader yields
`predator_indices=()`, so the per-predator distance tensor has a zero-width last axis
(`[.., 0]`). The behaviour-measure accumulators already guard on
`dist_pred.shape[1] > 0`, so predator-relative measures (e.g. M2_predator) come out
empty/NaN by design — expected and correct, because there is no predator to measure.
`load_env_params` does not crash (confirmed).
