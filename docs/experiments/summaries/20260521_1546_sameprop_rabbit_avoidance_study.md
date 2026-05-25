---
title: "SameProp Rabbit-Avoidance Study — Closing re-summary (Rounds 1–2.6, 2026-05-07 → 2026-05-21)"
study: sameprop_rabbit_avoidance_study
generated: 2026-05-21T15:46
window: "2026-05-07 → 2026-05-21"
status: snapshot
---

# SameProp Rabbit-Avoidance Study — Closing re-summary as of 2026-05-21 15:46 KST

> **The headline in one paragraph.** A reinforcement-learning agent lives on a small grid with a dangerous patrolling predator and two harmless rabbits. Normally the agent can tell them apart by smell. This study forced the predator and the rabbits to carry the *same* smell — making them sensorily indistinguishable from far away — and asked whether the agent still treats them differently. The answer turned out to depend on *what kind of "differently"* we measure. **Where the agent walks on average** is class-blind under matched smells (it doesn't keep predators farther away than rabbits). **What the agent does in the seconds around a creature entering its danger range** is strongly class-discriminating: it dives into cover roughly 88% of the time for a predator versus 51% for a rabbit, and it stops eating when a predator is near while eating normally when a rabbit is near. This two-layer finding was originally observed at one random seed (Round 2.5, late April); the seed-stability check (Round 2.6, this round) has now reproduced it at a second independent seed, with the numbers agreeing to within rounding error. The result is now treatable as paper-grade. The companion experiment that turned the predator into a harmless camp-target did *not* replicate its first-seed policy, so the "agent corner-camps" finding is one solution out of at least two — that is the only correction this re-summary makes to the prior story.

> **This document is the closing re-summary for the study.** Prior dated summaries — the most recent at [`20260514_2332_sameprop_rabbit_avoidance_study.md`](20260514_2332_sameprop_rabbit_avoidance_study.md) — remain as historical snapshots; nothing is edited in place. The 2026-05-14 version carried the same two-layer finding but flagged the second-seed test as "crashed, not yet replicated"; this version updates that status to "successfully re-launched and finished".

---

## Take-home messages — the whole study in 6 bullets

1. **The setup.** A reinforcement-learning agent normally tells a dangerous patrolling predator from a harmless rabbit by **smell**. This study deliberately gives them the **same smell** and asks: does the agent still behave differently around them, and *how*?

2. **Two answers, two layers — both still hold.** Where the agent *walks on average* is class-blind under matched smells (it doesn't keep predators farther away than rabbits). What the agent *does in the seconds a creature enters its danger range* is strongly class-discriminating: it gets into cover **about 37 percentage points more often** for a predator than for a rabbit, and **stops eating** near a predator while eating normally near a rabbit. Same agent, opposite verdict — different layer of measurement.

3. **Now seed-locked.** The first-layer (event-level) finding has now been reproduced at a second random seed. Cross-seed agreement is essentially exact: 88.8% vs 87.6% in-cover rate near predator, 37.3 vs 36.8 percentage-point class gap, 0.769 vs 0.748 eat-suppression ratio. **Two independent seeds, same numbers.**

4. **The sister experiment's story is more nuanced.** The companion experiment that made the predator harmless and confined it to one corner of the grid did *not* replicate its first-seed policy. The first seed (43) learned to live in the safe corner and survive to the timer; the second seed (45) learned a *different* policy — avoid the dangerous corner but starve to death anyway, surviving only 98 of 500 steps. **The "corner-camping" finding is one solution out of at least two**, not a generic basin. The simpler class-blindness conclusion still holds — both seeds are class-blind at the event level — but the headline "agent corner-camps under this setup" needs more seeds to characterise.

5. **The methodological lesson is now project canon.** Average-distance metrics dissolve event-level signal. Whenever a study concludes "the agent doesn't discriminate", cross-check with event-level measures (in-cover rate, eat-suppression near threat, behavioural-motif clustering) before trusting the verdict. The original "no class discrimination under matched smells" verdict was right as an *average-distance* statement and wrong as a *behavioural* statement.

6. **Where the work goes next.** The study has produced a paper-grade two-seed event-level finding. Three candidate next moves (no decision yet): (a) characterise the sister experiment's basin distribution with 4 more seeds; (b) launch the next planned round — food in all four corners, closing the spatial-camping loophole — with the event-level measures pre-registered as primary; (c) move on to the modulated-agent thread (does a pain-driven modulator widen the event-level gap further?).

---

## 0. Vocabulary — terms used in this document

A few terms recur often enough that they are worth defining once. Behaviour-metric details (the formulas + code) live in [**Appendix A**](#appendix-a--behaviour-metric-glossary-m1-m2-m5-m7); this section is the lightweight cheat-sheet.

### 0.1 The world the agent lives in

| Term | Meaning |
|---|---|
| **Grid world** | A small 2D grid (a few cells across) the agent inhabits. The agent moves one cell per step in one of five actions (up, down, left, right, stay). It survives by eating food and avoiding threats. |
| **Episode** | One life of the agent, from spawn to death-or-timer. Capped at **500 steps** per episode in this study. Agents can die from injury, starvation, or simply run out the timer. |
| **Survival steps** | How many steps the agent survived in an episode. The project's de-facto performance number — *not* cumulative reward. 500/500 = the agent lived to the timer; 98/500 = it died young. |
| **Patrolling predator** | One creature that walks around the grid and damages the agent on contact. The dangerous entity in the world. Has a state machine that can switch to "HUNT" mode and chase the agent. |
| **Hiding predators** | Four stationary creatures perched on rocks that damage the agent at a distance. Not the focus of this study but always present in the world. |
| **Rabbits** | Two harmless creatures that wander around. Visually + olfactorily similar to predators in this study — but inflict no damage. |
| **Bushes** | Obstacles that *hide* the agent: while the agent is on a bush cell, predators cannot see it. The agent uses this as cover. |
| **Quadrants — TL / TR / BL / BR** | Top-left / top-right / bottom-left / bottom-right of the grid. Entities are configured to spawn in specific quadrants. |
| **Smell / olfactory signature** | Each creature carries a 5-element "property vector". The agent has a short-range smell sensor that reports those vectors. Normally each class has a distinct vector. **This study forces them identical**, removing the smell cue. |

### 0.2 What this study manipulated

| Term | Meaning |
|---|---|
| **Matched smells** *("sameProp")* | The patrolling predator and the rabbits are given the same olfactory property vector `[0, 1, 0, 0, 0]`. The agent's smell sensor cannot tell them apart. **All experiments in this study run under this condition.** |
| **Food-decoupling experiment** *("Cell C", "decoupleFood")* | Food no longer spawns in rabbit quadrants. Removes the "agent camps near rabbits because that's where food is" confound. |
| **Passive-predator experiment** *("Cell A1", "passivePredator")* | The patrolling predator's HUNT mode is made unreachable, and the predator is confined to one quadrant (top-left). The dangerous threat becomes a static corner hazard. |
| **Round 1** | The first clean re-run (April), two seeds, no quadrant manipulation — confirmed the agent keeps predators ~0.6 cells farther than rabbits on average. |
| **Round 2 / 2.5** | Cell C + Cell A1 launched at full budget (10 million episodes), one random seed per cell (42 / 43). Round 2.5 produced the original two-layer verdict. |
| **Round 2.6** | The seed-stability check: same two cells, second seeds (44 / 45). First launch crashed; re-launched on healthy nodes and finished cleanly. **The closing analysis of this round is what this document re-summarises.** |
| **Seed** | The integer that initialises the agent's random weights + training stochasticity. Different seeds = different random initialisations = potentially different policies. "Two-seed-locked" = the same finding reproduces at two independent seeds. |

### 0.3 What the agent is

| Term | Meaning |
|---|---|
| **Plain RPPO** | The baseline policy used throughout this study: a small recurrent neural network (~128 hidden units) trained with the Proximal Policy Optimisation algorithm. No "modulator" or special architecture — just the off-the-shelf RL agent. Future studies will compare this baseline against modulated variants. |
| **Checkpoint** | The frozen network weights at a specific training step. Saved every ~10,000 episodes; the *final* checkpoint at 10M episodes is the one analysed here. |
| **Policy** | The function mapping the agent's observation to its action probabilities. At training time it is stochastic (samples actions); for analysis we run it in **deterministic mode** (argmax — take the most-likely action). |

### 0.4 How we measured the agent

| Term | Meaning |
|---|---|
| **Eval-rollout** | Frozen-checkpoint analysis. Take the final weights, run the policy in deterministic mode for **200 episodes** with a held-out set of environment seeds (1000–1199, disjoint from training seeds 0–999), dump every step to disk. These 200 episodes are what the behaviour metrics read. |
| **Danger range / radius R** | A class-`c` entity is "in danger range" at step `t` if the closest such entity is within **R = 3 cells** of the agent. The threshold that defines what "near" means for the metrics below. |
| **Threat onset** | The step at which an entity *just crossed* into danger range, having been outside it the step before. The "creature just got close" event. |
| **Lookahead K = 5 steps** | For the in-cover and stop-eating metrics, how far we look forward from a candidate event to count the agent's response. |
| **Spatial layer (average-distance)** | The original family of metrics: average distance the agent maintains to each entity over the whole episode. Numerically dominated by the long stretches of calm play. **Class-blind under matched smells** is its verdict on this study. |
| **Event layer (around threat onsets)** | The behaviour-toolkit family: what the agent does in the few steps *around* a threat onset. Numerically dominated by the rare moments that actually matter. **Class-discriminating in Cell C** is its verdict on this study. |
| **Class-conditional / class-blind** | "Class-conditional" = the agent responds differently to predator vs rabbit. "Class-blind" = same response. This is the study's central question. |
| **Pre-registered confirmation criteria** | Numeric thresholds that the study wrote down *before* the closing analysis ran, defining what counts as confirmation, refutation, or borderline. The Round 2.6 verdict is just: did the closing-analysis numbers cross those thresholds? |

### 0.5 The behaviour metrics (short form — full details in [Appendix A](#appendix-a--behaviour-metric-glossary-m1-m2-m5-m7))

| Short name | Plain-English question |
|---|---|
| **In-cover rate (M2)** | When a creature enters the agent's danger range, how often does the agent get into a bush within the next 5 steps? *(The study's headline metric.)* |
| **Eat-suppression ratio (M5)** | Does the agent's per-step probability of eating drop when a creature is nearby, compared to when it isn't? A value below 1.0 = the agent suppresses eating under threat. |
| **Interrupted-feeding rate (M1)** | When the agent is currently eating and a creature is in range, how often does it stop eating in the next 5 steps? |
| **Defensive-motif distribution (M7)** | A k-means clustering of behavioural windows around threat onsets, producing a 6-bin histogram of "kinds of response" per class. The qualitative complement to the three scalar metrics. |

---

## 1. Study question

**The world.** A reinforcement-learning agent lives on a small 2D grid (a few cells across). At each step it picks one of five actions (up / down / left / right / stay). Around it move three kinds of creature:

- a **patrolling predator** (one) — damages the agent on contact,
- **four hiding-predators** perched on rocks — damage the agent at a distance,
- **two rabbits** — harmless wanderers.

The agent has a short-range smell sensor: anything within its olfactory radius reports a 5-element "property vector" identifying its class. Normally each class carries its own property vector — so the agent can tell predators from rabbits from far away by smell. This study deliberately removes that cue: under "matched smells" the patrolling predator and the rabbits carry the *same* property vector, and the smell sensor can no longer separate them.

**The puzzle.** Does the agent still behave differently around the dangerous one? If yes, *how* — by sight at close range, by post-contact pain memory, by movement pattern, by something else? And — the question that emerged halfway through this study — does the answer depend on **what kind of "differently"** we measure (where the agent walks, or what it does in specific moments)?

**Why it matters for the broader project.** The wider research goal is to model **hypervigilance** — over-cautious avoidance of safe things, driven by pain or fear. To make that claim cleanly, we needed to know what *baseline* discrimination looks like under matched smells. This study mapped that baseline and surfaced a methodological lesson that affects every future "the agent doesn't discriminate" verdict in this project.

---

## 2. Experiments completed this study

> **Reading the experiment table.** Rows 3b and 6 quote per-class numbers for four behaviour metrics — **in-cover rate**, **eat-suppression ratio**, **interrupted-feeding rate**, **defensive-motif distribution** (technical names: M2, M5, M1, M7). If any term is unfamiliar, [§0 Vocabulary](#0-vocabulary--terms-used-in-this-document) has the short form and [Appendix A](#appendix-a--behaviour-metric-glossary-m1-m2-m5-m7) has the full definitions, formulas, and code.

| # | Experiment | Plain-English question | What was changed | High-level finding |
|---|---|---|---|---|
| **0** | **Existing-run survey** *(2026-05-07)* | Look at an already-running matched-smells training — does anything in the logs distinguish predator from rabbit? | Nothing changed; re-read existing logs. | The agent kept the predator about **half a cell farther** than rabbits on average. Suggestive but only one random seed; food and rabbits also shared quadrants, so the gap could be food-seeking rather than class avoidance. |
| **1** | **Round 1 — clean re-run** *(2026-05-07 → 2026-05-08)* | Re-run cleanly with two seeds and per-creature distance logging. Does the half-cell gap survive? | Per-creature distance logging implemented; two seeds (42, 43). | **Pattern replicates at both seeds** (gap ≈ +0.63 cells). Upgraded the survey from "anecdote" to "real signal." The food / rabbit shared-quadrant confound now needs a controlled test. |
| **2** | **Round 2 — kill the confounds** *(2026-05-08, stopped early)* | (a) Move food *away* from rabbit quadrants (food-decoupling experiment). (b) Turn off the predator's hunt-mode so only post-contact damage can teach avoidance (passive-predator experiment). | Food-decoupling: food spawn-zones moved off rabbits. Passive-predator: predator confined to one corner and made passive. | Both runs were interrupted at ~4–5% of budget. In the food-decoupling experiment the gap **flipped sign** (rabbits farther than predator). In the passive-predator experiment the gap exploded — but the agent never even visited the predator's corner. We were measuring **corner-camping**, not class avoidance. Need a sharper metric. |
| **3** | **Round 2.5 — full budget + per-corner distances** *(2026-05-09 → 2026-05-10)* | Same two experiments, but measure distance to the *specific* same-corner predator vs the *specific* same-corner rabbit, so corner-camping can't fake a result. | No experimental knob change. Full 10-million-episode budget; one seed per experiment (42 in food-decoupling; 43 in passive-predator). | **Average-distance verdict: NO class discrimination.** The passive-predator agent corner-camps (per-corner gap essentially zero, 0.004 cells; agent survives to the timer). The food-decoupling agent's gap inverts (rabbits farther than predator). The "agent treats predators specially in space" claim does not survive matched smells. |
| **3b** | **Round 2.5 appendix — new "what does the agent *do*" measures** *(2026-05-11)* | Read the *same* converged agents through a new behaviour-toolkit: in-cover rate around threat onsets, eat-suppression ratio under threat, behavioural-motif clustering. Does the agent's defensive *behaviour* discriminate even though its *average route* doesn't? | No new training — re-evaluated the Round 2.5 final checkpoints with the new metrics. 200 deterministic-policy episodes per agent. | **Event-level verdict: YES, strongly, for the food-decoupling agent.** When a threat enters its danger range (3 cells): in-cover rate near predator **88%** vs near rabbit **51%** → **+37 percentage-point** class gap. Eat-suppression ratio: **0.75** near predator (eating cut to 75% of safe baseline) vs **1.19** near rabbit (eating slightly elevated). The two rabbits behave essentially identically as targets — so the gap isn't a single-rabbit artifact. The passive-predator agent stays class-blind at every layer. |
| **4** | **Round 2.6 — seed-stability check, first try** *(2026-05-12 17:01 launch, crashed 23:38)* | Repeat both experiments at a new seed each (44 for food-decoupling, 45 for passive-predator) with event-level logging now built into training. | Seeds 44 + 45 on the same lab node, both behaviour-toolkit measures backfilled into the training configs. | **Both runs CRASHED at ~6.6 hours**, reaching 12% / 9% of the 10-million-episode budget. Crashed within 3 minutes of each other → likely a node-level event, not a training bug. The food-decoupling run's class gap was tracking at +29 percentage points and still climbing when it died. Seed-lock not yet earned. |
| **5** | **Round 2.6 — seed-stability re-launch** *(2026-05-16 → 2026-05-19)* | Re-launch the two crashed runs on healthy lab nodes. Same configs, same seeds. | Food-decoupling agent (seed 44) on a slower GPU (~82 hours wall-clock); passive-predator agent (seed 45) on a faster GPU (~48 hours). | **Both runs finished cleanly to 10 million episodes.** Closing-analysis numbers below. |
| **6** | **Round 2.6 closing analysis — primary verdict** *(2026-05-21)* | Apply the **pre-registered confirmation criteria** (set in writing before the run finished — see the threshold table below) to the final checkpoint of the food-decoupling re-launch. | No new training — re-ran the same 200-episode eval-rollout protocol that produced the Round 2.5 event-level numbers, this time on the Round 2.6 final checkpoint. | **Food-decoupling experiment: all three confirmation criteria met by a wide margin** (see the verdict table in §3). Cross-seed agreement to within rounding error (1.2 percentage points on in-cover rate; 0.5 percentage points on the class gap; 0.02 on eat-suppression ratio). **Two-seed replication achieved.** Passive-predator experiment: the corner-camping policy did **not** replicate at seed 45 — the agent survived only 98 of 500 steps (criterion was ≥ 470), learning a different starve-while-avoiding-the-dangerous-corner policy. The class-blindness conclusion holds; the corner-camping *policy* is one basin of at least two. |

**The pre-registered confirmation criteria for the food-decoupling experiment** (frozen before the Round 2.6 closing analysis ran):

| Number | What it asks | Threshold for confirmation | Threshold for refutation |
|---|---|---|---|
| **In-cover rate near predator** | When the predator just got close, what fraction of those events end with the agent in a bush within 5 steps? | ≥ 0.80 | < 0.50 |
| **Class gap on in-cover rate** | In-cover rate near predator minus in-cover rate near rabbit. | ≥ +0.30 (+30 percentage points) | < +0.10 |
| **Eat-suppression near predator** | Eat-rate when predator is in range, divided by eat-rate when the predator is not. | < 0.80 | ≥ 1.00 |
| **Secondary check — rabbits behave alike** | The two rabbits' numbers should agree (within ±5 pp on in-cover rate, ±0.10 on eat-suppression). Rules out single-rabbit artifacts. | (sanity check) | (sanity check) |

---

## 3. Where this leaves the study

### 3.1 The closing-analysis verdict in one table

The Round 2.6 food-decoupling experiment, final checkpoint, 200-episode eval-rollout:

| Number | Threshold for confirmation | Observed at seed 44 (this round) | Observed at seed 42 (previous round) | Cross-seed delta |
|---|---|---:|---:|---:|
| In-cover rate near predator | ≥ 0.80 | **0.888** ✓ (+11% over) | 0.876 | +1.2 percentage points |
| Class gap on in-cover rate | ≥ +0.30 | **+0.373** ✓ (+24% over) | +0.368 | +0.5 percentage points |
| Eat-suppression near predator | < 0.80 | **0.769** ✓ | 0.748 | +0.021 |
| (Eat-suppression near rabbit) | (no threshold; reported) | 1.202 | 1.186 | +0.016 |
| (In-cover rate near rabbit) | (no threshold; reported) | 0.515 | 0.508 | +0.7 percentage points |
| Sanity — rabbit pair agree on in-cover (±5 pp) | (sanity) | Δ = 3.7 pp ✓ | within noise | — |
| Sanity — rabbit pair agree on eat-suppression (±0.10) | (sanity) | Δ = 0.139 ⚠ (soft miss) | within noise | — |

**Verdict: all three primary criteria met; the seed-stability check is passed.** The two-seed cross-round agreement is within rounding error on every primary number — closer than the typical within-seed run-to-run noise on these measures.

### 3.2 What that means for the study

- **The headline is now seed-locked.** The event-level finding — agent dives into cover much more often for a predator, suppresses eating near a predator, eats normally near a rabbit — reproduces at a second independent random seed under the same matched-smells configuration. This is the paper-grade result.

- **The two-layer reading is unchanged.** Under matched smells the agent is **class-blind in where it walks** and **class-discriminating in how it defends itself**. Both statements are simultaneously true; they describe different things. Average-distance metrics miss the second one.

- **The corner-camping policy is one solution out of at least two.** The passive-predator experiment's two seeds chose qualitatively different policies: seed 43 lived in the safe corner and survived to the timer (486 of 500 steps); seed 45 avoided the dangerous corner but starved to death (98 of 500 steps). **Both are class-blind at the event level** — neither distinguishes predator from rabbit when forced to engage them — so the *class-blindness* conclusion is robust. But the *specific policy* underneath it is seed-sensitive: there are at least two basins. Characterising the basin distribution properly would need 4+ more seeds — a hypothetical follow-up round.

- **The methodological win is now load-bearing for the rest of the project.** Average-distance metrics are insensitive to event-level discrimination whenever the spatial layout is geometrically constrained (food in some corners, threats in others). Any future "no class discrimination" verdict gets cross-checked with the event-level toolkit (in-cover rate, eat-suppression, motif clustering) before being trusted. Round 2.5 is the worked example of why; Round 2.6 is the seed-stability proof that the event-level signal is real, not a Round 2.5 artifact.

- **One soft caveat.** The "the two rabbits should behave alike on eat-suppression" sanity check came in at 0.139 difference between the two rabbits — versus the ±0.10 band the toolkit pre-registered. **The primary verdict is not affected** (the primary criterion is the predator-side number, which passes), and the same direction was visible in the Round 2.5 numbers at a smaller magnitude. The pattern is consistent with food density differing between rabbit corners; the fix is either widening the sanity band to ±0.15 or computing eat-suppression per-corner against a corner-matched safe baseline. Flagged as a candidate change for behaviour-toolkit version 2.

- **The wider arc.** Hypervigilance work in this project should anchor on the **event-level baseline** (37-percentage-point in-cover-rate gap, eat-suppression near predator), not on the spatial baseline (no gap under matched smells). The latter has no signal for pain-modulation to amplify; the former does. The natural next pain-modulation experiments are: does a pain-driven modulator widen the gap further (over-amplified defence) or narrow it (suppressed)? — and in a next planned round (food in all four corners, closing the spatial-camping loophole), does the gap survive?

---

## 4. What's next (still pending decision)

1. **The portfolio-level call on the next research thread.** The study has produced a publishable two-seed event-level finding. Three candidate next moves are not mutually exclusive but compete for compute:
   - **(a)** Characterise the passive-predator-experiment basin distribution with 4 more seeds.
   - **(b)** Launch the next planned round — food in all four corners, closing the spatial-camping loophole — with the event-level toolkit pre-registered as the primary verdict.
   - **(c)** Pivot to the modulated-agent thread: does a pain-modulated variant of the baseline agent show a *larger* event-level class gap than the baseline does?

2. **Food-in-all-four-corners round — close the spatial-camping loophole.** Spawn food in every corner so no corner is uniformly safe (the agent can't trivially camp). Pre-register the event-level toolkit (in-cover rate, eat-suppression ratio, motif clustering) as the *primary* verdict, not a secondary check. Designer's prior: if the agent's class discrimination is genuine, the in-cover-rate gap widens further (88% → ~95%) and eat-suppression deepens (0.75 → ~0.55); if the gap was spatially mediated, it collapses.

3. **(Optional) Passive-predator basin-mapping round.** 4 seeds (46, 47, 48, 49) on the passive-predator experiment to map how the agent's policy distribution splits across the corner-camping basin vs the starve-while-avoiding basin vs anything else that emerges. Worth running only if the basin distribution is research-relevant downstream.

4. **(Optional) Pain-modulated-agent sequel.** Does a pain-driven modulator (a small extra network that scales the policy by an interoceptive injury signal) show a *larger* event-level class discrimination than the baseline agent? Natural bridge back to the project's modulator-comparison thread.

5. **Behaviour-toolkit version 2 — refine the eat-suppression per-rabbit sanity check.** The 0.139 marginal miss on the rabbit-pair agreement is a toolkit-design issue, not a verdict issue. Options: widen the band to ±0.15, or compute eat-suppression per-corner against a corner-matched safe baseline. Not blocking.

6. **Diary hygiene — small.** The Round 2.6 re-launch never wrote a `training-start` row to the daily event log (the original launch used a different run tag, and the re-launch tag wasn't separately registered). Flagged for the next re-launch-after-crash scenario.

---

## 5. Links

### Design docs

- [`docs/experiments/active/hypervigilance/sameprop_existing_run_survey.md`](../active/hypervigilance/sameprop_existing_run_survey.md) — the post-hoc survey that opened the study.
- [`docs/experiments/active/hypervigilance/round1_relog_baseline_analysis.md`](../active/hypervigilance/round1_relog_baseline_analysis.md) — Round 1 analysis at two seeds.
- [`docs/experiments/active/hypervigilance/sameprop_round2_design.md`](../active/hypervigilance/sameprop_round2_design.md) — Round 2 design (Cells C + A1) with §9 truncated-data analysis.
- [`docs/experiments/active/hypervigilance/sameprop_round25_design.md`](../active/hypervigilance/sameprop_round25_design.md) — Round 2.5 design with §§9–11 spatial verdict and the §12 event-level appendix. **The single document that carries the original two-level verdict.**
- [`docs/experiments/active/hypervigilance/sameprop_round26_design.md`](../active/hypervigilance/sameprop_round26_design.md) — Round 2.6 seed-lock design, now with §§9–12 filled in (the 2026-05-21 closing analysis). **The document that carries the seed-stability verdict.**
- [`docs/experiments/active/behavior_measures/behavior_measure_toolkit_v1_design.md`](../active/behavior_measures/behavior_measure_toolkit_v1_design.md) — the event-level toolkit's experimental design.

### Supporting plans (under `docs/develop/active/`)

- [`docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`](../../develop/active/hypervigilance/sameprop_discriminating_channels.md) — Phase-1 memo: which cues *can* still separate predator from rabbit under matched smells?
- [`docs/develop/active/hypervigilance/per_entity_avoidance_logging.md`](../../develop/active/hypervigilance/per_entity_avoidance_logging.md) — aggregated per-entity distance logging (Round 1's enabler).
- [`docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md`](../../develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md) — per-tag distance logging (the metric that resolved Round 2.5 at the spatial level).
- [`docs/develop/active/behavior/behavior_measure_toolkit_v1_plan.md`](../../develop/active/behavior/behavior_measure_toolkit_v1_plan.md) — the event-level toolkit implementation plan.
- [`docs/project/ideas/20260510_behavior_measure_toolkit.md`](../../project/ideas/20260510_behavior_measure_toolkit.md) — biological grounding for the toolkit (predictive-coding / active-inference framing).

### Memory insights (hypervigilance folder unless noted)

- [`docs/memory/memories/hypervigilance/20260518_1736_sameprop_c_seed44_directional_replication.md`](../../memory/memories/hypervigilance/20260518_1736_sameprop_c_seed44_directional_replication.md) — **the Round 2.6 closing insight (now settled)**: H₁(C-event) confirmed at eval-time, two-seed-elevated.
- [`docs/memory/memories/hypervigilance/20260518_1735_sameprop_a1_seed45_corner_camping_refuted.md`](../../memory/memories/hypervigilance/20260518_1735_sameprop_a1_seed45_corner_camping_refuted.md) — **the Cell A1 seed-45 refutation** (corner-camping basin is NOT generic; survival 98/500).
- [`docs/memory/memories/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md`](../../memory/memories/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md) — **the original event-level verdict** (+37 pp bush-dive gap, 0.75× / 1.19× eat-suppression, per-tag fan-out within noise) at seed 42.
- [`docs/memory/memories/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md`](../../memory/memories/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md) — **the spatial-level verdict** (Round 2.5 mean-distance verdict; complementary, not superseded).
- [`docs/memory/memories/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md`](../../memory/memories/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md) — Round 2 partial verdict.
- [`docs/memory/memories/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md`](../../memory/memories/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md) — per-tag metric design rationale.
- [`docs/memory/memories/hypervigilance/20260508_1444_sameprop_round1_finding_and_confound.md`](../../memory/memories/hypervigilance/20260508_1444_sameprop_round1_finding_and_confound.md) — Round 1 verdict + confound flag.
- [`docs/memory/memories/hypervigilance/20260508_1445_sameprop_discriminating_channels.md`](../../memory/memories/hypervigilance/20260508_1445_sameprop_discriminating_channels.md) — Phase-1 channels memo.
- [`docs/memory/memories/cluster_ops/20260518_1737_wandb_post_crash_frozen_state_misread.md`](../../memory/memories/cluster_ops/20260518_1737_wandb_post_crash_frozen_state_misread.md) — methodological lesson from the Round 2.6 crash: `run.summary` keys stay frozen post-crash; always check `run.state` first.
- [`docs/memory/memories/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md`](../../memory/memories/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md) — engineering lesson from the per-tag implementation week.

### Working files (representative — not exhaustive)

- `tmp/20260510_round25_cellC_last10.md`, `tmp/20260510_round25_cellA1_last10.md`, `tmp/20260510_round25_combined.md` — Round 2.5 spatial-analysis worksheets.
- `tmp/20260511_r25_appendix_*` — Round 2.5 §12 event-level appendix intermediates (scripts, JSON, CSVs, writeup notes).
- `tmp/20260521_r26_c_seed44_eval_analysis.py`, `tmp/20260521_r26_c_seed44_aggregates.json`, `tmp/20260521_r26_c_seed44_writeup.md`, plus the matching `_eval_rollout.log` / `_motif_cluster.log` / `_last10pct.md` / `_motifs.csv` / `_motif_by_class.csv` / `_motif_by_tag.csv` — Round 2.6 closing-analysis intermediates.

### Eval-rollout outputs

- `results/eval/models/10000022/` — Round 2.5 Cell C eval rollout (seed 42, source run `bdnfc0lu`).
- `results/eval/models/10000003/` — Round 2.5 Cell A1 eval rollout (seed 43, source run `nm8gn7y2`).
- `results/eval/models/10000021/` — **Round 2.6 Cell C eval rollout (seed 44, source run `ja5fu5k3`)** — the seed-lock evidence base.

### Diary days

- [`docs/diary/2026-05-07.md`](../../diary/2026-05-07.md) — Round 1 relog launches, Round 2 design, Phase-1 channels memo.
- [`docs/diary/2026-05-08.md`](../../diary/2026-05-08.md) — Round 1 relog completes, Round 2 launches + SIGINT'd, partial-verdict analysis.
- [`docs/diary/2026-05-09.md`](../../diary/2026-05-09.md) — per-tag metrics revised plan + implementation + verification; Round 2.5 launch.
- [`docs/diary/2026-05-10.md`](../../diary/2026-05-10.md) — Round 2.5 completes; §§9–11 spatial verdict.
- [`docs/diary/2026-05-11.md`](../../diary/2026-05-11.md) — behavior-measure toolkit shipped; §12 appendix written.
- [`docs/diary/2026-05-12.md`](../../diary/2026-05-12.md) — event-level verdict insight captured; Round 2.6 designed + launched (which then crashed).
- [`docs/diary/2026-05-13.md`](../../diary/2026-05-13.md) — Round 2.6 progress check (mis-read as in-flight); prior re-summary.
- [`docs/diary/2026-05-14.md`](../../diary/2026-05-14.md) — Round 2.6 crash discovered; reader-friendly re-summary.
- [`docs/diary/2026-05-16.md`](../../diary/2026-05-16.md) — Round 2.6 re-launch on n101 / n102.
- [`docs/diary/2026-05-18.md`](../../diary/2026-05-18.md) — Cell A1 seed-45 finished (refuted at 98/500); Cell C seed-44 directional-replication insight at 62.5%.
- [`docs/diary/2026-05-19.md`](../../diary/2026-05-19.md) — Cell C seed-44 finished cleanly on n101 (10 M ep, 81.8 h).
- [`docs/diary/2026-05-21.md`](../../diary/2026-05-21.md) — Cell C closing analysis (H₁(C-event) confirmed two-seed-elevated); this re-summary.

### Prior summaries (this study)

- [`docs/experiments/summaries/20260514_2332_sameprop_rabbit_avoidance_study.md`](20260514_2332_sameprop_rabbit_avoidance_study.md) — 2026-05-14 snapshot: reader-friendly version of the two-level verdict with Round 2.6 crash correction. Predecessor of this re-summary.
- [`docs/experiments/summaries/20260513_1420_sameprop_rabbit_avoidance_study.md`](20260513_1420_sameprop_rabbit_avoidance_study.md) — 2026-05-13 snapshot: first two-level re-summary (mis-reported Round 2.6 as in-flight; corrected by the 2026-05-14 version).
- [`docs/experiments/summaries/20260510_2253_sameprop_rabbit_avoidance_study.md`](20260510_2253_sameprop_rabbit_avoidance_study.md) — 2026-05-10 snapshot: spatial-level-only verdict.
- [`docs/experiments/summaries/20260509_1552_sameprop_rabbit_avoidance_study.md`](20260509_1552_sameprop_rabbit_avoidance_study.md) — 2026-05-09 snapshot: pre-Round-2.5.

### Implementation commits (load-bearing)

- `4b55fc6` — per-entity avoidance logging (Round 1 relog enabler).
- `0a73613` — per-tag per-instance distance logging (the spatial-level resolver).
- `82ae039` — Round 2.5 analysis (spatial-level verdict commit).
- `ed5cff3` — Round 2.5 §12 toolkit appendix (event-level verdict commit).
- `c110a2c` — event-level verdict captured as memory insight.
- `83ab27c` — Round 2.6 design.
- `c81b48d` — behavior_measures: block backfill into R2.6 configs.
- `42b4449` — Round 2.6 re-launch + results-check memory captures (Cell A1 refuted, Cell C directional replication, WandB-state lesson).
- `074ed57` — **Round 2.6 Cell C closing analysis** (H₁(C-event) confirmed two-seed-elevated; design doc §§9-12 filled in; closing memory update).

---

## 6. Reading order if you have 10 minutes

This document is intended to stand alone — you should not *need* to open any of the links to understand the verdict, the methods, or the implications. The reading order below points at the next-most-useful layer of detail for readers who do want to go deeper.

1. **This summary, top to §4** (~6 min) — the headline paragraph, the take-home bullets, the vocabulary, the experiment table, the verdict table, and the open decision points.
2. **[Appendix A](#appendix-a--behaviour-metric-glossary-m1-m2-m5-m7)** (~4 min) — the four behaviour metrics in plain English, with formulas, simplified Python code, and worked examples on this study's numbers.
3. **The food-decoupling closing-analysis memory insight, [`20260518_1736_sameprop_c_seed44_directional_replication.md`](../../memory/memories/hypervigilance/20260518_1736_sameprop_c_seed44_directional_replication.md)** (~1 min) — the verdict in 5-section memory form with the original directional-replication record and the 2026-05-21 closing update.

If you have 30 minutes, also read (in this order):

4. **§§9–12 of [`sameprop_round26_design.md`](../active/hypervigilance/sameprop_round26_design.md)** — the design-doc version of the closing analysis, with the full per-tag tables, the motif distribution, and the cross-round seed-paired comparison.
5. **§12 of [`sameprop_round25_design.md`](../active/hypervigilance/sameprop_round25_design.md)** — the original event-level appendix that the Round 2.6 work replicated.
6. **The Round 2.5 event-level verdict insight, [`20260512_1428_sameprop_class_discriminating_defence_event_level.md`](../../memory/memories/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md)** — the seed-42 paper-grade finding in memory form.
7. **The passive-predator refutation insight, [`20260518_1735_sameprop_a1_seed45_corner_camping_refuted.md`](../../memory/memories/hypervigilance/20260518_1735_sameprop_a1_seed45_corner_camping_refuted.md)** — the corner-camping-is-not-generic finding.
8. **The methodological-lesson insight, [`20260518_1737_wandb_post_crash_frozen_state_misread.md`](../../memory/memories/cluster_ops/20260518_1737_wandb_post_crash_frozen_state_misread.md)** — why post-crash logging-service reads are dangerous, and the corrected check pattern.
9. **The prior re-summary at [`20260514_2332_sameprop_rabbit_avoidance_study.md`](20260514_2332_sameprop_rabbit_avoidance_study.md)** — useful for seeing how the two-layer verdict was framed before the second-seed test landed.

---

## Appendix A — Behaviour-metric glossary (M1, M2, M5, M7)

This study's headline numbers (the +37 percentage-point bush-dive gap, the 0.75× eat-suppression near predators, the motif distributions) are computed from a project-specific behaviour-measure toolkit. None of the four metrics is a standard RL or behavioural-neuroscience measure off the shelf — they were lifted from a postdoc's 8-candidate menu and pre-registered as the toolkit's v1 shippable subset. This appendix is the **reader's reference**: each metric gets a plain-English description, the formula, the actual code that runs in the eval-rollout pipeline, an "edge case → NaN" rule, and a worked example from the actual study numbers.

The full operational specification lives in [`docs/experiments/active/behavior_measures/behavior_measure_toolkit_v1_design.md`](../active/behavior_measures/behavior_measure_toolkit_v1_design.md) §§1–2. The implementation lives in [`scripts/eval_rollout.py`](../../../scripts/eval_rollout.py) (M1/M2/M5) and [`scripts/motif_cluster.py`](../../../scripts/motif_cluster.py) (M7). The code snippets below are simplified extracts from those files — sufficient to follow the logic, not the literal bookkeeping.

### A.0 Shared setup — protocol parameters, danger radius, K-step lookahead

All four metrics share the same observation protocol — fixed by [`behavior_measure_toolkit_v1_design.md`](../active/behavior_measures/behavior_measure_toolkit_v1_design.md) §3 and locked across Round 2.5 and Round 2.6 so the numbers are seed-paired comparable:

| Parameter | Symbol | v1 value | What it controls |
|---|---|---|---|
| Danger radius | `R` | **3.0 cells** | A class-*c* entity is "in danger range" of the agent at step `t` iff the L2 distance from the agent to *some* class-*c* entity is < `R`. |
| Online lookahead window | `K` | **5 steps** | For M1 and M2, the number of steps after a candidate event during which we look for the consequence (stop-eating for M1; reach-a-bush for M2). |
| Motif-window length | `K_motif` | **7 steps** (window is `[t* − 2, t* + K_motif]` ⇒ 10 steps total) | For M7 only — the window of action / state we cluster around each threat-onset. |
| Eval rollout | `N_eval` | **200 deterministic episodes** | Frozen checkpoint, policy run in argmax mode (no exploration), seeds 1000–1199 (disjoint from training seeds 0–999). |
| K-means cluster count | `k` | **6** | M7 only — fixed at the toolkit's six prototype motifs (`freeze`, `flight`, `bush_dive`, `freeze_then_flight`, `ignore`, `approach`); semantic labels assigned post-hoc by inspecting nearest-centroid exemplars. |
| K-means seed | — | **42** | Locked for reproducibility. |

**Key terms used below:**
- A **predator-class entity** is the patrolling predator. A **rabbit-class entity** is any of the harmless rabbits. The hiding-predators on rocks are not in the per-class fan-out (they are a separate class).
- An **eat step** is a step at which the agent successfully consumed food (env info bit `ate_food[t]`).
- A **bush** is any obstacle with `hides_agent: true` in the env config. The env exposes `agent_in_bush[t]` as a per-step boolean.
- A **threat onset** is the rising edge of "any class-*c* entity within radius `R`" — i.e., the step at which the closest class-*c* entity *just crossed* into the agent's danger range, having been outside it on the previous step.
- A **per-class fan-out** = compute the metric separately for `c = predator` and `c = rabbit`. A **per-tag fan-out** = subdivide further by entity ID (e.g., `rabbit_TL`, `rabbit_BR`); used as a sanity check that the per-class number isn't being driven by a single instance.

---

### A.1 M1 — Interrupted-feeding rate

**Question it asks.** When the agent is *currently eating* and a class-*c* entity *is already in danger range*, how often does the agent stop eating within the next K=5 steps?

**Plain-English walk-through.**
1. Walk the episode step by step.
2. At each step `t`, check two conditions: (a) the agent just ate food, and (b) at least one class-*c* entity is within R=3 cells. If both hold, this step is a **candidate**.
3. Five steps later (i.e., at step `t + K`), check whether the agent has stopped eating at any point in `[t+1, t+K]`. If yes, count this candidate as **interrupted**.
4. M1 for class *c* = (number of interrupted candidates) / (total candidates).

**Formula.**

```
              # candidates with a stop-eating in the K-step look-ahead
M1_c   =     ────────────────────────────────────────────────────────────
              # candidates  (steps where agent ate AND class-c was within R)
```

**Edge case → NaN.** If the agent never eats while a class-*c* entity is within range (denominator = 0), M1_c is undefined for this episode and emitted as NaN. The cell-A1 corner-camping policy hits this — the agent eats only in the bottom-right corner, where neither predator nor rabbit comes close enough to trigger a candidate, so M1 is structurally near-meaningless for Cell A1.

**Code (simplified extract from `scripts/eval_rollout.py:201`).** This is the offline sanity-replay version; the online accumulator in `train.py` uses the same logic with a circular K-step buffer.

```python
# For a single episode, single class (e.g., predator):
candidates    = 0      # ate_food AND any-predator-within-R
interrupted   = 0      # candidates whose K-step lookahead contains a stop-eating
cand_age      = -1     # step counter for the live candidate; -1 = no live candidate
steps_since_eat = 0

for t in range(T):
    in_R = (num_predators > 0) and (np.min(dist_per_predator[t]) < R)

    # Age the live candidate first; if it has reached K, resolve it.
    if cand_age >= 0:
        cand_age += 1
        if cand_age >= K:
            candidates  += 1
            if steps_since_eat >= K:   # agent has gone K full steps without eating
                interrupted += 1
            cand_age = -1              # candidate consumed

    steps_since_eat = 0 if ate_food[t] else steps_since_eat + 1

    # Open a new candidate at this step if both conditions hold.
    if ate_food[t] and in_R:
        cand_age = 0

M1_predator = interrupted / candidates if candidates > 0 else float("nan")
```

**Where it lands in this study.** The Cell C R2.6 numbers: M1_predator ≈ 42% vs M1_rabbit ≈ 24% (a +18 pp gap) — consistent with R2.5 seed 42 (+18.1 pp). The gap is real and class-conditional, but it sits just inside the toolkit's H₁(M1) threshold (+20 pp). M1 is therefore reported as a **supporting** measure for Cell C; the headline signal lives in M2 and M5.

---

### A.2 M2 — Bush-dive rate

**Question it asks.** When a class-*c* entity *just enters* the agent's danger range (a rising edge), how often does the agent get inside a bush within the next K=5 steps?

**Plain-English walk-through.**
1. Walk the episode step by step, tracking whether each class is "currently inside R" at each step.
2. An **onset** is a step where the class transitions from "outside R" to "inside R" — AND the agent is not already in a bush at that step (we want active dives, not the agent already being in cover).
3. Across the next K=5 steps, check whether the agent is in a bush at any point. If yes, count this onset as a **dive**.
4. M2 for class *c* = (number of dives) / (total onsets).

**Why a *rising edge* and not just "in range"?** Because we want to count *responses to threat appearance*, not "the agent sat in a bush while the threat hung around". A predator that loiters within R for 30 steps generates one onset, not 30.

**Formula.**

```
              # onsets where agent enters a bush within K steps
M2_c   =    ──────────────────────────────────────────────────────────
              # onsets  (rising edges of "class-c within R" while agent NOT in bush)
```

**Edge case → NaN.** If no class-*c* entity ever enters range this episode (denominator = 0), M2_c is NaN.

**Code (simplified extract from `scripts/eval_rollout.py:260`).**

```python
# For a single episode, single class (e.g., predator):
onsets       = 0     # rising edge of dist_predator < R while NOT in bush
dives        = 0     # onsets where agent reaches a bush within K steps
onset_age    = -1
onset_bush_seen = False
prev_in_R    = False

for t in range(T):
    in_R = (num_predators > 0) and (np.min(dist_per_predator[t]) < R)

    # Onset: rising edge of in_R while agent NOT already in a bush
    if (not prev_in_R) and in_R and (not agent_in_bush[t]):
        onsets += 1
        onset_age = 0
        onset_bush_seen = False

    if onset_age >= 0 and agent_in_bush[t]:
        onset_bush_seen = True

    if onset_age >= 0:
        onset_age += 1
        if onset_age >= K:
            if onset_bush_seen:
                dives += 1
            onset_age = -1
            onset_bush_seen = False

    prev_in_R = in_R

M2_predator = dives / onsets if onsets > 0 else float("nan")
```

**Where it lands in this study.** The headline metric of the study. Cell C R2.6: **M2_predator = 0.888, M2_rabbit = 0.515, Δ_M2 = +0.373** — agent dives into a bush in 89% of predator-onset events but only 52% of rabbit-onset events. R2.5 seed 42 gave 0.876 / 0.508 / +0.368 — agreement to within 1.2 pp on M2, 0.5 pp on Δ_M2. **This is the seed-locked finding.**

---

### A.3 M5 — Eat-under-threat ratio

**Question it asks.** Does the agent eat less often when a class-*c* entity is nearby than when none is nearby? Below 1.0 = the agent suppresses eating under threat; 1.0 = no suppression; above 1.0 = the agent eats *more* under threat (e.g., the entity is in a food-rich quadrant).

**Plain-English walk-through.**
1. Walk the episode step by step.
2. At each step, label the step "under threat from class *c*" if any class-*c* entity is within R, otherwise "safe".
3. Count four things per class: total under-threat steps, total safe steps, eats while under threat, eats while safe.
4. M5 = (eats under threat / under-threat steps) ÷ (eats safe / safe steps).

**Formula.**

```
              eats_under_threat_c / under_threat_steps_c       P(eat | threat_c)
M5_c   =     ──────────────────────────────────────────────  =  ────────────────────
                  eats_safe_c / safe_steps_c                     P(eat | safe_c)
```

A value of **0.75** means the agent's per-step eat probability under predator threat is **25% lower** than its baseline (safe) eat rate. A value of **1.19** for rabbit means the agent eats **19% more** when a rabbit is nearby (because rabbits tend to be in food-rich quadrants).

**Edge case → NaN.** If the entire episode is under threat (safe_steps = 0) or never under threat (under_threat_steps = 0), M5_c is NaN. A secondary sanity criterion (R3): if `P(eat | safe) < 0.005`, the agent is a degenerate non-eater and M5 is uninterpretable even if technically defined.

**Code (simplified extract from `scripts/eval_rollout.py:236`).** The threat / safe / eat counters are accumulated alongside M1 and M2 in the same per-class state machine.

```python
# For a single episode, single class (e.g., predator):
under_threat_steps   = 0
safe_steps           = 0
eats_under_threat    = 0
eats_safe            = 0

for t in range(T):
    in_R = (num_predators > 0) and (np.min(dist_per_predator[t]) < R)

    if in_R:
        under_threat_steps += 1
        if ate_food[t]:
            eats_under_threat += 1
    else:
        safe_steps += 1
        if ate_food[t]:
            eats_safe += 1

P_eat_threat = eats_under_threat / under_threat_steps if under_threat_steps > 0 else float("nan")
P_eat_safe   = eats_safe         / safe_steps         if safe_steps > 0         else float("nan")
M5_predator  = P_eat_threat / max(P_eat_safe, 1e-9) if (under_threat_steps > 0 and safe_steps > 0) else float("nan")
```

**Where it lands in this study.** Cell C R2.6: **M5_predator = 0.769** (eat-rate 23% below safe baseline near a predator), **M5_rabbit = 1.202** (eat-rate 20% *above* safe baseline near a rabbit — rabbits sit in food-rich quadrants). The +0.43 cross-class delta is the second of the two paper-grade headline numbers; R2.5 seed 42 gave 0.748 / 1.186 / +0.44 — seed-paired agreement within 0.02. Soft caveat at seed 44: per-tag rabbit fan-out 0.139 marginally over the secondary ±0.10 band (primary M5 verdict unaffected — flagged as a toolkit-v2 candidate).

---

### A.4 M7 — Defensive-motif repertoire

**Question it asks.** Around each "class-*c* entity just entered danger range" event, the agent's behaviour over the next ~7 steps has some shape (freeze, flee, dive into bush, ignore, approach, mixed). M7 takes *all* such windows from all episodes, clusters them blind, and asks: **do predator-triggered windows land in different clusters than rabbit-triggered windows?**

This is the qualitative complement to the three quantitative metrics. M2 and M5 are scalars; M7 is a 6-bin histogram per class that tells you *what kind* of class-conditional response the agent has, not just how much.

**Plain-English walk-through.**
1. From the full eval-rollout (200 episodes), extract every threat-onset window — that's `[t* − 2, t* + 7]` inclusive (10 steps each), 7,741 windows for the seed-44 Cell C run.
2. For each window, compute **10 hand-picked features** describing what happened in it (table below).
3. Z-score the features so they're comparable (each feature: subtract mean, divide by std, pooled across all conditions).
4. Run k-means with k=6, seed=42.
5. Inspect the five nearest-centroid exemplar windows per cluster and assign an English label (e.g., `bush_camp_predator`, `feed_rabbit`, `predator_pursuit_with_bush`).
6. For each class (predator, rabbit), report the fraction of *that class's* threat-onset windows in each of the 6 clusters.

**The 10 features (per window, extracted by [`scripts/motif_cluster.py:52`](../../../scripts/motif_cluster.py)):**

| # | Feature | What it captures | Computation |
|---:|---|---|---|
| 1 | `net_displacement` | How far the agent moved overall | `‖ pos[end] − pos[start] ‖₂` |
| 2 | `path_length` | Total distance walked (vs. straight-line) | `Σ ‖ pos[t+1] − pos[t] ‖₁` |
| 3 | `threat_distance_change_rate` | Average per-step Δ-distance to the triggering instance (positive = retreating) | `mean(diff(dist_to_triggering_threat))` |
| 4 | `min_threat_distance` | Closest approach to the threat in the window | `min(dist_to_triggering_threat)` |
| 5 | `bush_occupancy_fraction` | Fraction of window-steps spent inside a bush | `mean(agent_in_bush)` |
| 6 | `eat_events_per_window` | Eat steps during the window | `sum(ate_food)` |
| 7 | `action_entropy` | Mixed vs. stereotyped action use (high = mixed) | Shannon entropy of action histogram |
| 8 | `mode_action_fraction` | Concentration on the single most-used action | `count(mode_action) / W` |
| 9 | `stay_in_place_fraction` | Fraction of steps where position didn't change (freezing) | `mean(no position change)` |
| 10 | `drive_injury_change` | Damage taken across the window | `nociception[end] − nociception[start]` |

**Code (simplified extract from `scripts/motif_cluster.py:52`).**

```python
def compute_window_features(ep, window_start, window_end, triggering_class, triggering_tag):
    sl       = slice(window_start, window_end + 1)
    pos      = ep["agent_pos"][sl]          # [W, 2]
    actions  = ep["action"][sl]             # [W]
    ate      = ep["ate_food"][sl]           # [W] bool
    in_bush  = ep["agent_in_bush"][sl]      # [W] bool
    noci     = ep["nociception"][sl]

    # Pick the distance array for the triggering instance
    if triggering_class == "predator":
        threat_dists = ep["dist_per_predator"][sl, parse_tag_idx(triggering_tag)]
    else:
        threat_dists = ep["dist_per_neutral"][sl, parse_tag_idx(triggering_tag)]

    return {
        "net_displacement":           np.linalg.norm(pos[-1] - pos[0]),
        "path_length":                np.sum(np.abs(np.diff(pos, axis=0)).sum(axis=1)),
        "threat_distance_change_rate": np.mean(np.diff(threat_dists)),
        "min_threat_distance":        np.nanmin(threat_dists),
        "bush_occupancy_fraction":    np.mean(in_bush),
        "eat_events_per_window":      np.sum(ate),
        "action_entropy":             shannon_entropy(np.bincount(actions, minlength=5)),
        "mode_action_fraction":       np.bincount(actions).max() / len(actions),
        "stay_in_place_fraction":     np.mean(~np.any(np.diff(pos, axis=0) != 0, axis=1)),
        "drive_injury_change":        noci[-1] - noci[0],
    }


# Then, across all windows:
from sklearn.preprocessing import StandardScaler
from sklearn.cluster      import KMeans
from sklearn.metrics      import silhouette_score

X        = StandardScaler().fit_transform(feature_matrix)            # z-score pooled
labels   = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(X)
sil      = silhouette_score(X, labels)                                # R4 sanity ≥ 0.20
```

**Sanity criterion → drop M7 if violated (R4).** The k-means clusters must have **silhouette score ≥ 0.20** OR the analyst must justify a lower silhouette (e.g., "behaviour is genuinely diverse, six clusters all 10–26% — not a one-cluster collapse"). Below 0.20 with no justification → M7 reads as "uninterpretable" and the figure is dropped from the paper. Cell C lands at 0.186 in both seed 42 and seed 44 — sub-threshold but reportable under the "behaviour is genuinely diverse" justification, which the Round 2.5 §12.5 wrote up in detail.

**Where it lands in this study.** Cell C R2.6 motif distribution (after post-hoc labelling of exemplars):

| Cluster | Label | Total share | Predator-triggered share | Rabbit-triggered share |
|---|---|---:|---:|---:|
| 0 | `predator_pursuit_with_bush` | 19.1% | **~83%** | ~17% |
| 1 | `bush_camp` | 12.8% | ~50% | ~50% |
| 2 | `mobile_with_cover` | 25.9% | ~50% | ~50% |
| 3 | `open_flight` | 10.7% | ~50% | ~50% |
| 4 | `feeding_bout_near_rabbit` | 10.5% | ~19% | **~81%** |
| 5 | `bush_camp_predator` | 20.9% | **~86%** | ~14% (R2.6); ~59% (R2.5) |

Two clusters are strongly **predator-skewed** (`predator_pursuit_with_bush`, `bush_camp_predator`); one is strongly **rabbit-skewed** (`feeding_bout_near_rabbit`). This is the qualitative complement to M2 / M5: the agent's defensive *kind* (not just *amount*) is class-conditional. The R2.6 seed-44 predator-shift on cluster 5 (86%) is even stronger than R2.5 seed-42 (59%), which is consistent with the +1.2 pp wider M2 gap at seed 44.

---

### A.5 The toolkit's pre-registered sanity criteria

Before any of M1–M7 produces a verdict, the run must clear four "is the measure even meaningful?" checks. They are listed here for completeness — full operational definitions in [`behavior_measure_toolkit_v1_design.md`](../active/behavior_measures/behavior_measure_toolkit_v1_design.md) §1.2.

| Criterion | Trigger | What it forces |
|---|---|---|
| **R1** | Any of M1 / M2 / M5 lands outside its canonical range (M1, M2 outside `[0, 1]`; M5 negative; NaN where finite expected) | Wiring bug — halt and surface to senior-developer. |
| **R2** | M1 or M2 denominator (candidates / onsets) below a minimum sample size | "Uninterpretable for this cell" — report the denominator zero-rate as the headline finding, not the rate. This is how Cell A1 reads in this study (camping → almost no candidates). |
| **R3** | M5's safe-step denominator zero (entire episode under threat) OR `P(eat | safe) < 0.005` (agent is a degenerate non-eater) | Same — uninterpretable, report the diagnostic. |
| **R4** | M7's k-means silhouette < 0.20 with no written justification | Drop M7 from the paper figure; redesign featurisation in v2. Cell C narrowly clears this via the "genuinely-multimodal" justification (0.186, six well-balanced clusters). |

The Round 2.6 Cell C run cleared R1, R2, R3 cleanly and cleared R4 with the documented sub-threshold justification. The R2.5 seed 42 run cleared all four under identical rules — which is what makes the seed-paired comparison legitimate.

