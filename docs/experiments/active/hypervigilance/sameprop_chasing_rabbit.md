---
title: "SameProp Round 4 — chasing rabbit: does the agent flee a harmless animal that actively chases it?"
topic: hypervigilance
status: active
created: 2026-05-29
last_updated: 2026-06-02
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

### Results (deterministic eval, checkpoint 9.52M — read this first)

**One-paragraph plain-English headline.** We took the agent trained alongside one
dangerous predator and two harmless chasing rabbits, and rolled it out 200 times in a
world where **every dangerous entity was deleted** — no patrolling predator, no corner
ambushers — leaving only the two harmless chasing rabbits. We compared this against a
matched 200-rollout in the **original predator-present world**, using the **same saved
brain** (the checkpoint taken ~95% of the way through training, at 9.52 million
episodes), the **same greedy "always pick the best action" policy**, and the **same
200 random-world seeds**, so the two result sets are directly comparable to each other.
**Result: when nothing can hurt it, the agent drops its defensive behaviour.** It
survives every single episode (200/200 vs 121/200 with the predator), forages roughly
50% more food (about 115 vs 77 food items per episode), lets the rabbits come ~25%
closer (nearest rabbit averages 1.4 cells away vs 1.8 with the predator), and stops
suppressing eating when a rabbit is nearby. Two further findings: (a) **even with the
predator present, the agent already tells the predator and the rabbit apart** despite
their identical smell and identical chasing motion — it keeps the predator ~4.3 cells
away while letting the chasing rabbit to ~1.8 cells, and dives for cover far more often
for the predator; and (b) **removing the predator relaxes the agent's caution toward
the rabbits too**, even though the rabbits themselves did not change. Together this says
the agent's threat discrimination is **driven by what has actually damaged it** (a
learned per-animal danger memory read off the visual "what class is this" channel), not
by chasing-motion or proximity. **Caveat up front:** this is one seed, one architecture,
and a checkpoint at ~95% of training (not the final converged model) — clean *within*
this matched protocol, but still a single model.

**Translating the metric shorthand used below.** *Survival steps* = how many timesteps
the agent stays alive out of a 500-step episode cap (our performance measure; we never
use cumulative reward). *Termination reason* = why an episode ended: `1` = survived to
the 500-step cap, `2` = died of injury (predator/hazard damage), `4` = starved.
*Nearest-X distance* = average grid-cell distance the agent keeps from the closest
entity of class X. *M2 bush-dive rate* = of the times a threat closes in on the agent
(an "onset"), the fraction where the agent responds by diving into a bush it can hide
in — our main active-avoidance measure. *M5 eat-under-threat ratio* = how the agent's
eating rate when a threat is nearby compares to when it is safe; **below 1 means it
suppresses eating under threat** (cautious), **near/above 1 means it eats just as freely
near the threat** (unbothered). *M1 interrupted-feeding rate* = fraction of feeding
attempts the agent aborts when a threat appears.

#### R.1 Matched comparison table

All values are deterministic-eval aggregates over the 200 fixed-seed episodes on the
same 9.52M checkpoint. Numbers reproduced independently from the per-episode `.npz`
trajectories and the `online_replay.json` behaviour-measure dumps; see the working
file `tmp/20260601_153000_noPredator_transfer.md` for the extraction script.

| Measure (plain meaning) | With predator | No predator |
|---|---|---|
| **Survival mean** (steps out of 500) | 403.2 | **500.0** |
| Survival spread (std / min) | +/-140.2 / 23 | +/-0.0 / 500 |
| **% episodes surviving to the 500-step cap** | 60.5% (121/200) | **100% (200/200)** |
| **Deaths** (episodes ending in injury or starvation) | 79 | **0** |
| -- death breakdown (injury `2` / starvation `4`) | 23 / 56 | 0 / 0 |
| Mean nearest-**predator** distance (cells) | 4.26 (+/-1.20) | -- (no predator) |
| Mean nearest-**rabbit** distance (cells) | 1.79 (+/-0.39) | **1.41 (+/-0.27)** |
| -- per-rabbit (top-left tag / bottom-right tag) | 2.41 / 2.43 | 2.04 / 2.02 |
| Fraction of steps a rabbit is within 3 cells | 0.79 | 0.83 |
| **Food eaten per episode** | 76.8 (+/-36.4) | **114.8 (+/-5.5)** |
| Rabbit contacts per episode (all harmless) | 110.5 | 213.1 |
| Predator contacts per episode | 6.45 | 0.0 |
| **M2 bush-dive -- PREDATOR** (defensive dive rate) | 0.565 (1270 dives / 2249 onsets) | NaN (no predator) |
| **M2 bush-dive -- RABBIT** | 0.409 (833 / 2039) | **0.159 (401 / 2518)** |
| **M5 eat-under-threat -- PREDATOR** | 0.675 (eats *less* near predator) | NaN |
| **M5 eat-under-threat -- RABBIT** | 1.975 | **1.076** (eats ~as freely) |
| **M1 interrupted-feeding** (either class) | 0.000 | 0.000 |

Notes on the table. (1) The per-rabbit distances (2.0-2.4 cells) are larger than the
*nearest*-rabbit distance (1.4-1.8) because "nearest" takes the closer of the two
rabbits at each step; both views agree the rabbits sit closer in the no-predator world.
(2) The behaviour-measure toolkit's own logged within-3-cells fraction reads 0.754
(predator world) / 0.806 (no-predator world) -- same direction as the 0.79 / 0.83 we
recomputed here, ~0.04 lower, a denominator-convention difference that does not move the
conclusion. (3) Every "predator" measure in the no-predator column is NaN/empty by
design: with the predator deleted the per-predator distance tensor has a zero-width axis
and the accumulators correctly skip it (the parity check in the methods table above).

#### R.2 What the agent does when nothing can hurt it

In the no-predator world the agent **stops treating the chasing rabbits as threats**,
on every channel we can read:

- **It never dies** (0 / 200) and **survives the full episode every time** (100% reach
  the 500-step cap), versus 79 deaths and only 60.5% full-length survival when the
  predator is present. The death-cause split with the predator (23 injury, 56
  starvation) shows the predator world kills the agent both directly (injury) and
  indirectly -- starvation, because staying defensive costs foraging time.
- **It forages straight through the rabbits.** Food intake jumps from ~77 to ~115 items
  per episode (+50%), and the variance collapses (+/-36 -> +/-5) -- once nothing can hurt
  it, every episode becomes a clean, near-identical foraging run.
- **It lets the rabbits in closer** (nearest-rabbit 1.79 -> 1.41 cells; within-3-cells
  occupancy 0.79 -> 0.83) and **stops suppressing eating near them** (M5 rabbit ratio
  1.975 -> 1.076, i.e. from clearly-cautious down to essentially-unbothered).
- **It never aborts a meal** for a rabbit (M1 = 0.000 in both worlds -- interrupted
  feeding was never a rabbit-driven behaviour here).

**The residual rabbit bush-dive (M2 = 0.159) is incidental foraging, not defense.** We
tested this directly rather than asserting it. If the 0.159 were active defense, bush
entries should cluster when a rabbit is *closing in* (small distance). They do the
opposite: in the no-predator world the agent enters a bush when the nearest rabbit is
**1.73 cells away on average -- farther than its 1.41-cell overall average**, and only
20% of its steps are spent in bushes. In the predator world the sign flips: bush entries
happen when the nearest rabbit is **closer than average** (1.47 vs 1.75 overall), the
agent spends **twice as much time in bushes** (44% of steps), and it **suppresses eating
while hidden** (eats on 6.5% of in-bush steps vs 16% in the no-predator world). The
timing of bush entry relative to threat proximity reverses between the two worlds -- that
is the signature of defensive diving in the predator world and of ordinary
move-through-a-bush-while-foraging in the no-predator world. The residual 0.159 is the
latter.

#### R.3 The matched comparison: the agent discriminates by damage, not motion

Reading the two columns against each other answers the thread's central question.

- **With the predator present, the agent already separates the predator from the
  chasing rabbit** even though the two share matched smell *and* now share the same
  chasing motion. It keeps the predator 4.26 cells away but the rabbit only 1.79; it
  dives for cover for the predator at 0.565 vs 0.409 for the rabbit; and it suppresses
  eating near the predator (M5 0.675, below 1) while eating freely near the rabbit (M5
  1.975, well above 1). With smell and motion held identical across the two animals, the
  only cue left to discriminate on is the **visual class channel** (a predator reads on
  a different one-hot channel than a neutral animal) paired with a **learned association
  that the predator-class animal is the one that has damaged it**.
- **Removing the predator relaxes caution toward the rabbits too** -- generalized
  vigilance. The rabbits did not change between the two worlds, yet rabbit bush-dive
  falls 0.409 -> 0.159, rabbit eat-suppression lifts (M5 1.975 -> 1.076), and the agent
  lets the rabbits ~0.4 cells closer. So part of the caution the agent *appeared* to
  show toward the rabbits in the training world was actually driven by the predator's
  presence (a generalized "threats are around" posture), not by the rabbits themselves.

Both points push the same way: the agent's defensive discrimination is **damage-driven**
-- a learned, per-animal danger memory keyed to the visual "what class is this" channel
-- **not** driven by chasing motion (the rabbits chase exactly like the predator and are
still not feared) or by raw proximity. This is consistent with the earlier rounds: in
Round 3 randomizing the predator's chase parameters did not collapse the predator-rabbit
gap, and in this chasing-rabbit run making the rabbits chase did not open a new gap.

#### R.4 Measure-validity payoff (the original worry, retired for this world)

The concern that launched this whole thread was that our main avoidance measure -- the
bush-dive rate (M2) -- might be **blind to flight**: if the agent avoided rabbits by
*running away* rather than diving into cover, M2 would under-count rabbit-avoidance and
mislead us. The two independent channels -- the **distance/flight channel** (how far the
agent keeps each animal) and the **bush-dive channel** (M2) -- **agree in both worlds**.
Where the agent is cautious (predator), both the distance is large and M2 is high; where
it is relaxed (rabbits, and especially the no-predator world), both the distance is
small and M2 is low. There is no hidden flight-based avoidance that the bush-dive
measure missed. For this world, M2 was **not** misleading -- it tracked the flight read.

#### R.5 Watch the rollouts

- No-predator eval video (the agent foraging unbothered through the chasing rabbits):
  `results/eval/noPredator_chasingRabbit/models/9520028/videos/eval_9520028.mp4`
- With-predator matched eval video (defensive behaviour for comparison):
  `results/eval/withPredator_chasingRabbit/models/9520028/videos/eval_9520028.mp4`
  *(rendering at time of writing -- the `videos/` directory exists but the `.mp4` may
  not be present yet; check before linking.)*

#### R.6 Conclusions and honest caveats

**Conclusion.** Dropping every damaging entity from the world causes the trained agent
to **abandon its defensive behaviour** -- it survives perfectly, forages freely, and lets
the harmless chasing rabbits right up to it. The matched predator-present comparison
shows the agent had already been discriminating the predator from an
identically-smelling, identically-chasing rabbit, and that part of its apparent
rabbit-caution was a spillover of general predator-driven vigilance. The most
parsimonious account across this thread: **threat discrimination in this matched-smell
world is a learned per-animal danger association (damage-driven), read off the visual
class channel -- not a chasing-motion or proximity heuristic.** The two avoidance
channels (distance and bush-dive) agree, so the bush-dive measure was valid here.

**Caveats, stated plainly.**
1. **Single seed.** One training seed (42); the eval replays 200 *world* seeds against
   that one brain, which controls eval noise but not training-init variance. A
   multi-seed repeat is needed before any published claim.
2. **Checkpoint at ~95%, not final.** The brain evaluated here is the 9.52M-episode
   checkpoint (~95% of the 10M budget), not the converged final model. The direction is
   unlikely to reverse (the predator-rabbit gap was already stable mid-training), but the
   absolute levels can still shift slightly.
3. **One architecture.** Recurrent-PPO only. Whether the damage-driven discrimination is
   architecture-general is untested.
4. **Clean within-protocol, still one model.** The two eval sets *are* directly
   comparable (same checkpoint, same greedy policy, same 200 seeds, byte-identical
   protocol) -- that is the strength of this comparison over the earlier online-preview.
   But "clean comparison between two rollouts of one model" is not the same as "robust
   across models," and conclusion strength is bounded by caveats 1-3.

_This Results block fills the pre-existing No-predator transfer eval subsection. The
parent design was lightly pre-registered ("watching, not gating"), so these conclusions
are observational rather than a thresholded confirm/refute._


## Matched-aggression control eval

**Why this eval exists (plain English).** In the original chasing-rabbit world the
predator picked its chase ability (how far it sees you, how long it can sprint, how
stubbornly it pursues) randomly each episode, while the two harmless rabbits always
chased with the *same fixed, aggressive* settings. So on any given episode the predator
could chase *less reliably* than the rabbits. That makes the agent's apparent
"avoid-the-predator-before-it-ever-touches-me" behaviour ambiguous: it could be real
danger inference, or just an artifact of the predator *looking or moving differently*
(an **approach-speed confound**). This control config removes that confound by setting
the predator's five chase parameters to be byte-identical to the rabbits' fixed values,
so predator and rabbit now differ in exactly one thing: the predator hurts on contact and
the rabbits do not. We then re-evaluate the already-trained chasing-rabbit model on it.

**Prediction.** If the agent truly cannot tell predator from rabbit from what it observes
(smell and chase dynamics are now identical, and the visual class channel is contact-only
since visual range is zero), then the pre-first-contact distance gap between predator and
rabbits should **collapse**, and any avoidance of the predator should appear **only after
the first painful contact** in an episode. A surviving pre-contact gap would instead say
the agent reads danger off the visual class channel on contact and generalises, not off
approach speed.

**Config + validation.** Eval config:
[`configs/experiment/hypervigilance/06-matchedAggression_chasingRabbit_eval.yaml`](../../../../configs/experiment/hypervigilance/06-matchedAggression_chasingRabbit_eval.yaml).
The only diff vs the training config (`04-sameProp_R4_chasingRabbit.yaml`) is the
predator's five chase fields, changed from per-episode `[low, high]` ranges to fixed
scalars equal to the rabbits' values. Loader validation confirmed: observation dimension
unchanged at **27** (breakdown byte-identical to config 04); `predator_indices=(0,)`,
`neutral_indices=(1,2)`, `animal_is_damaging=[True, False, False]`, `hunt_idx=(0,1,2)`;
all five chase fields equal across the three animals (each a degenerate `[v, v]` range:
detection `[10,10]`, stamina `[60,60]`, recovery `[1,1]`, hunt-threshold `[0.3,0.3]`,
lose-interest `[3,3]`); damage/nociception asymmetry preserved (predator `[15,45]` @ 0.9
vs rabbits `[0,0]` @ 0.1) and visual class channel 5 (predator) vs 7 (rabbits).

### Results — matched-aggression control (deterministic eval, final 10M checkpoint)

**One-paragraph plain-English headline (read this first).** The question this control
answers is sharp: *does the trained agent recognise and avoid the dangerous predator
**before** the predator ever touches it?* If it did, that would be impossible under our
own observation pipeline — at a distance the predator and the harmless rabbits are
indistinguishable (same smell, the agent's eyes have **zero range** so the "what class
is this" visual channel only fires on contact, and the agent's memory resets every
episode), so pre-contact the agent has **no information** about which animal is the
killer. Yet an earlier read showed the agent kept the predator about 2.5 cells farther
away than the rabbits *before any contact*, which looked like impossible pre-contact
discrimination. **This control shows that apparent gap was not real danger inference —
it was two stacked measurement artifacts.** When we remove both (see below), the
pre-contact distance gap between predator and rabbit collapses to **essentially zero,
and is in fact slightly negative** (the agent, if anything, sits a hair *closer* to the
predator than to a rabbit before contact). The clincher: in the matched world the agent
**dies in 91% of episodes** — if it could spot and flee the predator before contact it
would not be slaughtered when the predator beelines straight for it. The real
discrimination is **entirely post-contact**: predator touches the agent → pain → the
agent then keeps its distance. No hidden observation leak, no implementation bug. The
user's theoretical intuition — *matched cues ⟹ no pre-contact discrimination* — is
empirically vindicated.

**Translating the metric shorthand.** *Survival* = timesteps alive out of a 500-step
cap (our performance measure; never cumulative reward). *First-contact step* = the
timestep on which the predator first touches the agent in an episode; everything before
it is the **pre-contact window**, the only window in which the agent provably has zero
class information. *Pre-contact distance gap* = how much farther (in grid cells) the
agent keeps the predator than it keeps a rabbit, averaged over just the pre-contact
window — **this is the key evidence**; a positive gap means "predator held farther
(looks like avoidance)", ~zero or negative means "no pre-contact discrimination".
*M2 bush-dive rate* = of the times a threat closes in, the fraction where the agent
dives into a bush to hide (our main active-avoidance measure). *M5 eat-under-threat
ratio* = eating rate near a threat vs. when safe; **below 1 = suppresses eating
(cautious), above 1 = eats freely (unbothered)**.

#### The two arms (what differs)

Both arms are **deterministic** rollouts of the **same final converged checkpoint**
(10 million episodes, the `s42` chasing-rabbit model), over the **same 200 fixed eval
seeds**, so they are directly comparable to each other. The only difference is the
predator's chase behaviour:

- **Arm A — distributional predator** (the original training-world dynamics): the
  predator's five chase parameters (how far it detects the agent, how long it can
  sprint, how fast it recovers, how eagerly it re-engages, how stubbornly it pursues)
  are re-rolled randomly each episode, so on a given episode the predator can chase
  *less reliably* than the always-on rabbits.
- **Arm B — matched-aggression predator** (the control): the predator's five chase
  parameters are set **byte-equal to the rabbits'** fixed aggressive values (detect
  anywhere = 10, stamina 60, recovery 1.0, re-engage threshold 0.3, lose-interest 3.0).
  Now predator and rabbit are **behaviourally identical** and differ in exactly one
  thing — the predator deals damage on contact, the rabbits do not.

#### R.M1 Comparison table

All values are deterministic-eval aggregates over the 200 fixed-seed episodes on the
final 10M checkpoint. Numbers independently re-derived from the per-episode `.npz`
trajectories and the `online_replay.json` behaviour-measure dumps; extraction script and
the full reconciliation (the pre-contact gap reproduces to the digit under per-episode
averaging) are in the working file `tmp/20260602_061518_matchedAggression_control.md`.

| Measure (plain meaning) | Arm A — distributional | Arm B — matched-aggression |
|---|---|---|
| **Survival mean** (steps / 500) | 396.3 (±138.2) | **272.9 (±120.8)** |
| **% episodes reaching the 500 cap** | 53% (106/200) | **9% (17/200)** |
| **Deaths** (injury or starvation) | 94 | **183** |
| — death split (injury / starvation) | 33 / 61 | 76 / 107 |
| **Mean first-contact step** (lower = predator reaches agent sooner) | 85.2 | **33.3** |
| Episodes with no predator contact at all | 12 | 1 |
| **% steps predator within 2 cells** | 0.152 | **0.268** |
| M2 bush-dive — predator / rabbit | 0.542 / 0.416 | 0.699 / 0.592 |
| M5 eat-under-threat — predator / rabbit | 0.615 / 1.751 | 1.315 / 2.227 |

Reading the table: under matched aggression the predator reaches the agent **2.6×
sooner** (first contact 85→33), spends **~1.8× more time right next to it** (0.152→0.268
of steps within 2 cells), and the agent **dies far more** (survival 396→273; only 9% of
episodes survive the full 500 steps vs 53%). The agent is being run down. That alone is
hard to reconcile with any claim that it can see and flee the predator before contact.

#### R.M2 The key evidence — pre-contact distance gap, broken out by pooling method

This is the load-bearing table. Each cell is the **predator-minus-rabbit distance
averaged over the pre-contact window only** (positive = predator held farther =
*looks* like pre-contact avoidance). It is broken out four ways by **which rabbit
distance we compare against**, because the choice of comparison is exactly where the
first artifact hides.

| Pre-contact gap, predator vs… | Arm A — distributional | Arm B — matched-aggression |
|---|---|---|
| **nearest of the 2 rabbits** | **+2.52** | **+1.11** |
| **mean of the 2 rabbits** | +1.63 | **−0.15** |
| **rabbit_TL alone** (one rabbit) | +1.54 | **−0.24** |
| **rabbit_BR alone** (the other rabbit) | +1.71 | **−0.07** |

The whole story is in how the gap moves from the top row to the bottom rows, and from
Arm A to Arm B.

**Artifact 1 — "2 rabbits vs 1 predator" (a pooling bias, removed by comparing
1-vs-1).** The agent faces **two** rabbits but only **one** predator. "Distance to the
*nearest* rabbit" is the minimum of two independent draws, which is mechanically smaller
than the distance to the single predator — *even if the agent treats all three animals
identically*. So the +2.52-cell "nearest-rabbit" gap in Arm A is inflated purely by the
2-vs-1 geometry. Comparing the predator against **one** rabbit at a time (the `_mean`,
`_TL`, `_BR` rows) removes this bias and immediately shrinks the Arm-A gap from +2.52
down to ~+1.5 to +1.6.

**Artifact 2 — "the distributional predator chased less aggressively" (an
approach-speed confound, removed by matching chase parameters).** Even after fixing the
2-vs-1 bias, Arm A still shows a +1.5-to-+1.6 gap. But in Arm A the predator's chase was
randomly re-rolled, so on average it pursued *less reliably* than the always-on rabbits
— it approached more slowly and therefore simply *sat farther from the agent before
contact*, with no inference required. Matching the predator's chase parameters to the
rabbits' (Arm B) makes them approach identically (first-contact step drops 85→33), and
**this residual gap vanishes**: predator-vs-one-rabbit goes to **−0.15 / −0.24 / −0.07**
— zero, slightly negative.

**With both artifacts removed (Arm B, 1-vs-1): the pre-contact gap is ≈ 0, slightly
negative.** The agent does **not** keep the predator farther than a rabbit before the
first contact. There is **zero pre-contact discrimination** — exactly what the clean
(no-leak) observation pipeline predicts, since the agent has no class information in that
window.

#### R.M3 The clincher — it gets slaughtered under matched aggression

If the agent could recognise the predator and flee it pre-contact, matching the
predator's chase ability to the rabbits' should not be lethal — the agent would still
sidestep the predator the way it (appears to) avoid threats. Instead, under matched
aggression the agent **dies in 91% of episodes** (survival 272.9, only 9% reach the
500-step cap, 183/200 deaths), with the predator reaching it in 33 steps on average.
The agent cannot get out of the way of a predator it cannot distinguish until that
predator hurts it. This is direct behavioural confirmation that the avoidance is
**reactive (post-contact), not anticipatory (pre-contact)**.

The post-contact channel is exactly where discrimination *does* live and *should* live:
once the predator makes painful contact, the M2 bush-dive and M5 eat-suppression
machinery (and the kept-distance behaviour documented in the earlier sections of this
doc) kick in. Both arms keep the predator's M2 bush-dive rate above the rabbit's
(0.542 > 0.416 in Arm A; 0.699 > 0.592 in Arm B), and that separation is driven by
contact-triggered pain, fully consistent with the class-blind-at-distance observation.

#### R.M4 Watch the rollout

- Matched-aggression eval video (predator chasing exactly like the rabbits, agent run
  down repeatedly): `results/eval/matchedAggression_final/models/10000024/videos/eval_10000024.mp4`
  *(rendering at time of writing — the `videos/10000024/` directory exists but the
  `.mp4` may not be present yet; check before linking.)*

#### R.M5 Caveats (stated plainly)

1. **Single seed, one architecture.** One training seed (42), recurrent-PPO only. The
   eval replays 200 *world* seeds against that one brain — this controls eval noise but
   not training-init variance, and says nothing about architecture-generality.
2. **Arm B is a transfer eval (out-of-distribution).** The agent was trained with the
   *distributional* predator (Arm A dynamics) and is *evaluated* under matched
   aggression (Arm B), a world its policy never trained on. So Arm B's absolute levels
   (survival 273, 91% deaths) reflect a policy operating off-distribution and should not
   be read as "how good the agent is" — they are a stress test, not a performance score.
   **Crucially, the pre-contact-gap logic is immune to this caveat:** the argument
   concerns a window (before first contact) in which the agent provably has **zero class
   information** regardless of which distribution it trained on, so a ≈0 pre-contact gap
   there cannot be an artifact of the train/eval mismatch.
3. **Pre-contact gaps use per-episode-then-averaged pooling** (equal weight per episode),
   matching the convention used elsewhere in this doc. Pooling all pre-contact steps
   instead shifts magnitudes but not signs (the Arm-B-vs-one-rabbit gap stays ≤ ~+0.6 and
   the qualitative collapse holds); see the working file for both.

## Final synthesis — what we learned across the whole chasing-rabbit thread

**The single question, answered.** Does this matched-smell agent recognise and avoid the
dangerous predator *before* the predator ever touches it? **No.** Three independent lines
of evidence now converge on the same answer, and on the same mechanism — discrimination
is **post-contact and pain-driven**, never anticipatory:

1. **The observation audit** (`docs/reviews/chasingRabbit_obs_classLeak_audit.md`): at a
   distance the agent's senses are class-blind. Smell is matched across predator and
   rabbits; the visual "what class is this" channel only fires **on contact** (the
   agent's visual range is zero); and the recurrent memory **resets every episode**. So
   before the first contact, the agent has literally no signal that separates the killer
   from the harmless chasers. There is no observation leak.

2. **The matched-aggression control** (this section): once we strip out the two
   measurement artifacts that made pre-contact avoidance *look* real — the 2-rabbits-vs-1-
   predator pooling bias, and the slower-approaching distributional predator — the
   pre-contact distance gap **collapses to ≈ 0 (slightly negative)**. And under matched
   chase dynamics the agent is **run down and killed in 91% of episodes**, which it could
   not be if it were fleeing the predator pre-contact.

3. **The no-predator transfer eval** (earlier section of this doc): the agent's caution
   toward the rabbits is contingent on the predator being present and relaxes when every
   damaging entity is removed — i.e. the caution is a response to *actual danger
   experienced*, not a fixed reaction to chasing motion or proximity.

**The mechanism, stated once.** Threat discrimination in this world is a **learned
per-animal danger association keyed to painful contact** — predator touches the agent →
pain → the agent thereafter keeps its distance, dives for cover, and suppresses eating
near *that* class. It is **not** driven by chasing motion (the rabbits chase identically
and are not feared), **not** by smell (matched), and **not** by any pre-contact class
cue (there is none). Earlier rounds are consistent: randomising the predator's chase
(Round 3) did not collapse the predator-rabbit gap, and making the rabbits chase
(Round 4) did not open one.

**No bug.** The earlier "+2.5-cell pre-contact gap" was a real number computed correctly;
it just measured two confounds rather than danger inference, and both are now isolated and
removed. The observation pipeline behaves exactly as designed (class-blind at distance),
the GRU reset is working, and the agent's behaviour is exactly what that clean pipeline
predicts. The user's theoretical intuition — matched cues imply no pre-contact
discrimination — is empirically vindicated.

_This Results block fills the pre-existing Matched-aggression control subsection. The
parent design was lightly pre-registered ("watching, not gating"); the matched-aggression
control adds a **sharp, falsifiable** prediction (pre-contact gap should collapse under
matched chase dynamics) which the data confirms. Conclusions remain bounded by the
single-seed / single-architecture / transfer-eval caveats above._
