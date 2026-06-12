---
title: "Predator vs rabbit: does the agent recognise danger, or just react to pain?"
study: predator_rabbit_discrimination
generated: 2026-06-12T16:25
window: "2026-05-29 → 2026-06-12"
status: snapshot
---

# Predator vs rabbit: does the agent recognise danger, or just react to pain?

> This study asks a single question: when a harmful animal (a "predator", whose touch injures the agent) and a harmless one (a "rabbit", whose touch does nothing) are made **indistinguishable from the outside** — same smell, same chasing behaviour, same approach speed, same strike-and-retreat rhythm — and differ **only** in whether contact actually hurts, does the trained agent treat them differently *before* contact? Over a series of June experiments the answer converged to: **no, not in any anticipatory, danger-recognising way.** The agent's avoidance is **driven by the consequence of contact (pain/injury), not by recognising the predator at a distance.** When we made the harmless rabbits actively hunt the agent, the agent simply tolerated them and foraged through them. When we matched the predator's chase aggression to the rabbits', the agent could no longer stay ahead of it and died in 91% of episodes. When we removed the rabbits and left only the predator, the agent foraged and rested in place while the predator walked straight into it, hiding only *after* being hit. And the latest, cleanest training — a single predator and a single rabbit, byte-identical on every observable, where a single predator contact can now be lethal — still produced an agent that treats the two alike (class-blind). A second, equally important thread runs through the study: our **aggregate metrics initially hid a real conditional behaviour**, and reading individual trajectories step-by-step (plus the agent's raw observation vector) was what exposed the true mechanism — a methodology lesson that reshaped how we measure danger-discrimination in this project.

> This is the first reader-facing summary of the **June predator-vs-rabbit discrimination thread**. It builds on, but does not replace, the five earlier `sameprop_rabbit_avoidance_study` summaries (2026-05-09 → 2026-05-21), which cover the matched-smell rounds that preceded this work. This document is append-only — later re-summaries will be written as fresh dated files, not edits to this one.

## Take-home messages — the whole study in 6 bullets

1. **The setup.** We progressively stripped away every external difference between a harmful "predator" and a harmless "rabbit" — matching smell, chase behaviour, approach speed, and (in the final config) the strike-and-retreat rhythm — leaving danger (does contact hurt?) as the only thing separating them, then asked whether the agent avoids the dangerous one *before* contact.

2. **The headline finding.** The agent's avoidance is **pain-consequence-driven, not danger-recognising.** Across four controls — rabbits made to hunt, predator chase-matched to rabbits, predator-only world, and the final single-pred-vs-single-rabbit lethal-contact training — the agent never showed reliable *anticipatory* discrimination. It reacts to being hurt; it does not pre-empt the threat.

3. **Replication / seed status.** Single seed (42), single architecture (recurrent-PPO), evaluations on the 10-million-episode checkpoints. The damage-driven conclusion is consistent across several independent configs, but none of the June results is yet multi-seed — that hardening is still pending.

4. **Caveats / nuance.** The cleanest control (predator-only) is **out-of-distribution** — the agent trained with animals always present, never with zero rabbits — so its confidence is medium, not high. And an earlier May reading found the agent *was* discriminating at the **event level** (it dived into cover more often for the predator than the rabbit); the June controls reframe that as most likely a post-contact reaction rather than anticipation.

5. **The methodological lesson.** Our **aggregate statistics averaged away a real conditional behaviour.** A confidently-wrong "the agent cannot discriminate" conclusion survived several turns until we read individual episode trajectories step-by-step and inspected the agent's actual 27-number observation vector. That trajectory-level + raw-observation habit is now the project's default first move for behaviour questions, and is captured as a reusable analysis skill.

6. **Where it goes next.** With the new logging/checkpoint settings in place, the open moves are: (a) run the in-distribution confirmation (condition pre-contact avoidance on whether the rabbits are "accounted for"), (b) formally analyse the now-complete lethal-contact training at its final checkpoint, and (c) decide whether to give the agent a *learnable* route to danger recognition (e.g., a sensory channel or memory pressure) rather than concluding it simply cannot. No decision has been made.

## §0 Vocabulary — terms used in this document

### 0.1 What this study manipulated

| Term (shorthand) | Plain-English meaning |
|---|---|
| Predator | An animal whose touch injures the agent (`class: predator` → `is_damaging = True`). |
| Rabbit / neutral | The same kind of chasing animal whose touch does no harm (`class: neutral` → `is_damaging = False`). |
| Matched smell ("sameProp") | Predator and rabbit emit the **same** olfactory signature `[0,1,0,0,0]`, so smell cannot tell them apart at a distance. |
| Chasing rabbit (R4) | Experiment that gave the harmless rabbits the predator's hunting behaviour (they actively pursue the agent) while keeping them harmless — to test whether the *chasing motion* drives avoidance. |
| Matched aggression | A control where the predator's five chase parameters are set **equal** to the rabbits', removing any "the predator chases more weakly" confound. |
| Predator-only world | A control with the rabbits **removed**, leaving only the lone predator — the cleanest test of whether the agent pre-empts the predator. |
| Disengage-on-contact (strike-and-retreat) | A feature shipped this study: on contact an animal's energy drains to zero, so it must back off and recover instead of pinning the agent. Applied to **both** animals so the retreat rhythm is matched. |
| Lethal single contact | The final config draws the predator's per-touch injury from a wide band (5–120 on a 0–100 scale where 100 = death), so ~1 contact in 6 can kill outright. |
| Single-pred-vs-rabbit (cell 08) | The latest training: one predator + one rabbit, roaming the whole grid, identical on every observable, differing only in damage. The cleanest danger-discrimination contrast in the lineage. |
| Class-blind / class-discriminating | "Class-blind" = treats predator and rabbit the same; "class-discriminating" = behaves differently toward the two. |

### 0.2 How we measured the agent

| Term | Plain-English meaning |
|---|---|
| Survival steps | The project's headline performance measure — how many steps the agent stays alive. Never cumulative reward. |
| Eval-rollout | A frozen-checkpoint evaluation that replays the trained agent for N episodes with a deterministic (best-action) policy and records both summary measures and full step-by-step recordings. |
| Trajectory-level / story-level read | Reading individual episodes step-by-step ("what did the agent actually do?") instead of trusting averaged statistics. Now packaged as the `trajectory-story` analysis skill. |
| Visual channel (count) | The agent's vision only reports animals on its **own** cell, and reports them as a **count per class** (e.g. "2 rabbits on me"). Central to the methodology lesson — see Appendix A. |
| Pre-contact vs post-contact | Whether a defensive action happens *before* the animal touches the agent (anticipatory) or *after* (reactive). The whole study turns on this distinction. |
| M1 / M2 / M5 (behaviour measures) | Non-standard event-level metrics: M1 = interrupted-feeding rate, M2 = bush-dive rate, M5 = eat-under-threat ratio. Full definitions in Appendix A. |

### 0.3 The behaviour metrics (short form)

| Metric | Plain-English question | Full def |
|---|---|---|
| M1 (interrupted-feeding) | How often does the agent break off eating when a threat is near? | [Appendix A](#appendix-a--behaviour-measures--mechanisms-glossary) |
| M2 (bush-dive rate) | How often does the agent dive into cover when an animal approaches? | [Appendix A](#appendix-a--behaviour-measures--mechanisms-glossary) |
| M5 (eat-under-threat ratio) | Does the agent keep eating while a threat is close (>1) or suppress eating (<1)? | [Appendix A](#appendix-a--behaviour-measures--mechanisms-glossary) |

## §1 Study question

**The world.** The agent lives on a grid and must forage to survive: it moves, eats food that respawns in the quadrants, and rests to recover energy. Sharing the grid are *animals* — some harmful (predators) and some harmless (rabbits) — and *bushes* that conceal the agent (cover it can dive into). The agent senses the world through a 27-number observation: hunger, two pain signals (internal and external), a 5-way smell sensor, a 5-way collision sensor, its own body state, and an 8-way vision sensor. Crucially, with vision range set to zero, the agent only *sees* an animal's class when that animal is on its **own cell** — at a distance, the only thing it can sense about an approaching animal is its smell.

**The manipulation.** We made the predator and the rabbit identical on everything the agent can sense from afar — same smell, same chasing behaviour, same approach speed, same strike-and-retreat rhythm — so that the *only* difference between them is whether contact injures the agent. If the agent still behaves differently toward the two *before* contact, that difference can only come from the agent having **inferred** which one is dangerous.

**Why it matters.** This is the project's operational test of "pain-like" danger recognition. A hypervigilant, pain-shaped agent should learn to recognise and steer clear of a threat *before* it gets hurt — the way an animal that has been bitten avoids the thing that bit it. An agent that only reacts *after* being hurt is showing reflexive damage-avoidance, not the anticipatory, danger-recognising behaviour the project is trying to elicit. Distinguishing these two is exactly the construct-validity question for the whole hypervigilance line.

## §2 Experiments completed this study

> The table below references event-level behaviour metrics by short name (M1, M2, M5). Plain-English meanings are in [§0.3](#03-the-behaviour-metrics-short-form); full formulas, code, and worked numbers are in [Appendix A](#appendix-a--behaviour-measures--mechanisms-glossary).

| # | Experiment | Plain-English question | What was changed | High-level finding |
|---|---|---|---|---|
| 0 | Matched-smell rounds (prior anchor — May, *not run this window*) | Under identical smells, does the agent avoid the predator more than the rabbit? | Predator and rabbit share smell; food decoupled into a corner; 2-rabbit layout | **Two-level verdict**: spatially class-blind (equal mean distance) but class-discriminating at the *event* level (+37-point bush-dive gap), seed-locked across two seeds by 2026-05-21. |
| 1 | Chasing-rabbit (R4) | Does the *chasing motion* itself drive avoidance? | Gave the harmless rabbits the predator's hunting behaviour (they now pursue the agent) but kept them harmless | **No.** The agent tolerates the harmless chaser — lets it ride on its own cell, forages through it, never interrupts feeding for it. Motion is not the cue. |
| 2 | No-predator transfer | With all harmful entities removed, does the agent still treat the chasers as threats? | Removed every damaging entity; left only the 2 harmless chasers | **Ignores them entirely** — survives 500/500 episodes, lets rabbits approach to ~1.4 cells, eats 115 food/episode, interrupted-feeding (M1) = 0.000. |
| 3 | Matched-aggression | If the predator chases as reliably as the rabbits, can the agent stay ahead of it? | Set the predator's five chase parameters equal to the rabbits' | **No.** Survival drops to 273 steps; **91% of episodes end in death.** Once the predator chases as hard as the rabbits, the agent gets run down — its differentiation is post-contact. |
| 4 | Predator-only control | In the cleanest test, does the agent pre-empt the approaching lone predator? | Removed the rabbits; left only the patrolling predator | **No pre-emption.** The agent forages/rests in place while the predator beelines into it (first contact ~step 11) and hides only *after* being hit. Avoidance is post-contact / pain-reactive. |
| 5 | Single-pred-vs-rabbit, lethal contact (cell 08) | With the two animals byte-identical and a single predator contact now possibly lethal, does avoidance become anticipatory and class-specific? | One predator + one rabbit, identical on every observable + strike-and-retreat; predator damage 5–120 (≈1-in-6 lethal); rabbit harmless | **Class-blind.** The strike-and-retreat redesign fixed the rabbit-ride artifact and made avoidance robust, but the agent treats predator and rabbit alike — no danger-specific anticipation. (Training complete at ~10 M episodes; final-checkpoint formal analysis pending.) |

There were **no numeric pre-registered confirmation thresholds** for the June discrimination thread — these were exploratory controls, each designed to refute or confirm a specific mechanism (motion-driven? aggression-confound? anticipatory?), read qualitatively from trajectories plus the spatial/event measures. (The earlier matched-smell anchor *did* carry pre-registered thresholds; those live in the `sameprop_rabbit_avoidance_study` summaries.)

## §3 Where this leaves the study

### 3.1 The verdict in one table

| Claim under test | How it was tested | Result | Verdict |
|---|---|---|---|
| Avoidance is driven by the chasing *motion* | Chasing-rabbit (R4) + no-predator transfer | Agent tolerates harmless chasers; ignores them with predator removed (500/500 survival, M1 = 0.000) | **Refuted** — motion is not the cue |
| The predator/rabbit distance gap is real class recognition | Match predator chase aggression to rabbits; compare 1 predator to 1 rabbit | Gap collapses from +2.5 cells to −0.07…−0.24 cells; 91% death under matched aggression | **Refuted** — the gap was an aggression/number artifact |
| The agent pre-empts the predator (anticipatory avoidance) | Predator-only world, 20 episodes, trajectory read | Forages/rests in place until hit; hides only reactively; first contact ~step 11 | **Refuted (medium confidence)** — avoidance is post-contact / pain-reactive |
| A lethal, byte-identical predator makes avoidance anticipatory & class-specific | Single-pred-vs-rabbit lethal-contact training (cell 08) | Robust avoidance, but applied equally to predator and rabbit | **Class-blind** — danger not recognised at a distance |

**Verdict:** across every June control, the agent's defensive behaviour is best explained as **reaction to the consequence of contact (pain/injury), not recognition of danger before contact.** The one earlier signal of genuine discrimination — the event-level bush-dive gap under matched smells — most plausibly reflects post-contact reaction rather than anticipation, once the clean controls are taken into account.

### 3.2 What that means for the study

- **The headline is now well-supported (single-seed):** the agent is a pain-reactor, not a danger-recogniser. This is a meaningful *negative* construct-validity result for "anticipatory hypervigilance" under fully matched, sense-from-a-distance-impossible conditions.
- **There is a genuine cross-experiment pattern:** every time we removed a possible "tell" (motion, then aggression, then the rabbits themselves, then every observable difference), the apparent discrimination shrank toward zero. The discrimination was riding on confounds, not on danger recognition.
- **The main caveat is distributional.** The cleanest anti-anticipation result (predator-only) is out-of-distribution — the agent never trained with zero rabbits — so the in-distribution confirmation (condition pre-contact avoidance on whether the rabbits are "accounted for" in the agent's vision count) is the result that would settle it without the caveat. It has not been run yet.
- **The methodological surprise reshaped our process.** Aggregate statistics (mean distance, distance-matched flee rate of 76% vs 76%, M1 = 0.000) hid a real *conditional* behaviour because they mixed two regimes together and averaged the effect to zero. Reading individual trajectories and the raw observation vector exposed the true mechanism. This is now the default first move, packaged as the `trajectory-story` skill — see the metric-vs-story note at the end of this document.
- **A practical fix fell out of the work:** the disengage-on-contact (strike-and-retreat) feature, which removed the artifact where a harmless rabbit perpetually rides the agent's cell after contact, and tightened the predator↔rabbit behavioural match for all future runs.
- **The wider arc:** if the agent *cannot* recognise danger under these conditions, the next research question is whether that is a fundamental limit of the sensory/training setup or something a *learnable* route (a richer sensory channel, a memory/precision pressure, or a different objective) could change — which is where the hypervigilance line goes next.

## §4 What's next (still pending decision)

1. **Portfolio-level call — what is the next paper-shaped move?** Candidates:
   - (a) **Harden the negative result** — re-run the key controls at ≥3 seeds so "pain-reactor, not danger-recogniser" is multi-seed, and publish it as a clean construct-validity finding.
   - (b) **Give the agent a learnable route to recognition** — add a sensory or memory mechanism that *could* support pre-contact discrimination, and test whether anticipation emerges (turning the negative into a positive-direction study).
   - (c) **Pivot to the event-level thread** — the May matched-smell event-level discrimination is the one place a positive signal survived; deepen that instead.
2. **Run the in-distribution confirmation** (removes the out-of-distribution caveat on the anti-anticipation verdict): within the matched-aggression trajectories, condition the agent's pre-contact flee/cover response on whether both rabbits are currently "accounted for" in its vision count. If pre-emption is absent even when the count is full, post-contact-only avoidance is confirmed cleanly.
3. **Formally analyse the now-complete lethal-contact training** (cell 08, ~10 M episodes) at its final checkpoint with the trajectory + behaviour-measure protocol, and fill in the Results section of its design doc.
4. **Carry the new logging defaults into the next rPPO runs** — the sparser logging / 200k-episode checkpoint / keep-all-checkpoints settings are now the rPPO default, so the next discrimination runs will produce lighter, fully-retained checkpoints for re-analysis.

## §5 Links

**Design docs**
- [single_pred_rabbit_disengage](../active/hypervigilance/single_pred_rabbit_disengage.md) — the latest lethal-contact training (cell 08); Results section pending.
- [sameprop_chasing_rabbit](../active/hypervigilance/sameprop_chasing_rabbit.md) — the chasing-rabbit (R4) + no-predator + matched-aggression controls, with comparison tables.
- [sameprop_predator_distributional](../active/hypervigilance/sameprop_predator_distributional.md) — predator damage-distribution design context.

**Supporting plans / reviews**
- [DISENGAGE_ON_CONTACT](../../develop/active/env_entities/DISENGAGE_ON_CONTACT.md) — the strike-and-retreat feature plan.
- [disengage_on_contact_review](../../reviews/disengage_on_contact_review.md) — code review (APPROVE).
- [config_08_singlePredRabbit_disengage_preflight](../../reviews/config_08_singlePredRabbit_disengage_preflight.md) — pre-flight config audit.
- [chasingRabbit_obs_classLeak_audit](../../reviews/chasingRabbit_obs_classLeak_audit.md) — observation-leak audit (what the agent can sense about class).

**Memory insights**
- [20260609_1720_chasing_rabbit_avoidance_damage_driven](../../memory/memories/hypervigilance/20260609_1720_chasing_rabbit_avoidance_damage_driven.md) — the study verdict: avoidance is damage-driven, not motion-driven.
- [20260609_1747_avoidance_is_post_contact_not_preemptive](../../memory/memories/hypervigilance/20260609_1747_avoidance_is_post_contact_not_preemptive.md) — predator-only control: post-contact / pain-reactive, not anticipatory (supersedes 1719).
- [20260609_1721_aggregate_stats_hide_conditional_behavior](../../memory/memories/hypervigilance/20260609_1721_aggregate_stats_hide_conditional_behavior.md) — **the metric-vs-story methodology post-mortem** (the answer to "what was the problem?").
- [20260609_1719_predator_discrimination_visual_count_elimination](../../memory/memories/hypervigilance/20260609_1719_predator_discrimination_visual_count_elimination.md) — the visual-count "elimination" mechanism (factually correct; behavioural reading superseded).
- [20260512_1428_sameprop_class_discriminating_defence_event_level](../../memory/memories/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md) — the prior event-level discrimination anchor.
- [20260508_1445_sameprop_discriminating_channels](../../memory/memories/hypervigilance/20260508_1445_sameprop_discriminating_channels.md) — which observation channels carry class (vision/pain at contact only).
- [20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride](../../memory/memories/env_entities/20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride.md) — the rabbit-ride artifact that motivated the strike-and-retreat feature.

**Eval-rollout outputs / results**
- `results/eval/matchedAggression_traj/models/10000024/` — 200 recorded matched-aggression episodes (the trajectory + raw-obs read).
- `results/eval/{noPredator_chasingRabbit, matchedAggression_final, predatorOnly}/...` — the control evals (+ videos under each `videos/`).
- `results/JAX_RecurrentPPO/20260609-191226_recurrent_ppo_08-singlePredRabbit_disengage_s42/` — the completed lethal-contact training (~10 M episodes).

**Prior summaries (predecessor study)**
- [20260521_1546_sameprop_rabbit_avoidance_study](20260521_1546_sameprop_rabbit_avoidance_study.md) — matched-smell closing re-summary (event-level discrimination seed-locked); full M1/M2/M5 appendix.

**Diary days**
- [2026-06-09](../../diary/2026-06-09.md) — chasing-rabbit deep-dive, obs-leak audit, matched-aggression + predator-only controls, disengage feature shipped, four insights captured.

**Implementation commits (load-bearing)**
- `c67cc40` — `eval_rollout.py --record` (frozen-checkpoint eval emitting measures + recordings).
- `71a672f` — `trajectory-story` analysis skill + experiment-analyzer profile update.

## §6 Reading order if you have 10 minutes

This document is intended to stand alone — you should not need to open any of the links to understand the verdict, the methods, or the implications. The reading order below points at the next-most-useful layer of detail for readers who do want to go deeper.

1. **This summary** (~6 min) — the headline blockquote + the 6 take-home bullets + the §3.1 verdict table give you the whole study.
2. **[Appendix A](#appendix-a--behaviour-measures--mechanisms-glossary)** (~3 min) — the behaviour metrics (M1/M5), the visual-count mechanism, and the disengage feature, so the numbers in §2/§3 are verifiable.
3. **The metric-vs-story post-mortem** — memory insight [20260609_1721](../../memory/memories/hypervigilance/20260609_1721_aggregate_stats_hide_conditional_behavior.md) — the methodological heart of the study.
4. **The cleanest verdict** — memory insight [20260609_1747](../../memory/memories/hypervigilance/20260609_1747_avoidance_is_post_contact_not_preemptive.md) — the predator-only control.

If you have 30 minutes, also read the chasing-rabbit design doc ([sameprop_chasing_rabbit](../active/hypervigilance/sameprop_chasing_rabbit.md)) for the full control-by-control comparison tables.

## Appendix A — behaviour measures + mechanisms glossary

### A.0 Shared setup

| Constant | Value | Meaning |
|---|---|---|
| Observation dimension | 27 | Satiation(1) + intero-noci(1) + extero-noci(1) + olfaction(5) + collision(5) + proprioception(6) + visual(8) |
| Visual sensor range | 0 | Vision only reports animals on the agent's **own** cell |
| Predator visual channel | 5 | Vision one-hot index for a predator on the agent's cell |
| Neutral (rabbit) visual channel | 7 | Vision one-hot index for a rabbit on the agent's cell |
| Matched smell | `[0,1,0,0,0]` | Identical olfactory signature for predator and rabbit |
| Eval episodes | 20–200 | Deterministic (best-action) policy; per-episode carry reset |
| Checkpoint analysed | `10000024` (~10 M ep) | The frozen final-ish checkpoint used for the control evals |

**Key terms below:** *threat radius* = how close an animal is when a step counts as "under threat"; *bush-dive* = the agent moving into a concealing bush; *accounted for* = the agent's vision count shows all rabbits on its own cell.

### A.1 The visual channel is a COUNT (the mechanism behind the methodology lesson)

**Plain-English question.** What does the agent actually see when animals are on its cell?

**Walk-through.** Vision returns an 8-number vector. For each animal sitting on the agent's own cell, a one-hot at that animal's class-channel is added. Because they are *summed*, the channel holds a **count**: two rabbits on the agent's cell make channel 7 read `2.0`. This is why the agent could, in principle, discriminate by *elimination* — "both rabbits are on me (channel 7 = 2), so the third smell approaching must be the predator."

**Formula.**
```
visual[c] = sum over animals a on the agent's cell of  one_hot(class_channel[a], 8)[c]
class_channel = {predator: 5, neutral: 7}
```

**Edge case.** When the rabbits are scattered at a distance (not on the agent's cell), channel 7 = 0 — no elimination signal — and the agent reverts to "flee any approaching animal". Mixing the accounted (channel 7 full) and not-accounted (channel 7 = 0) regimes is exactly what averaged the conditional effect to zero in the aggregate stats.

**Code extract** (`src/environment/sensor.py`, `sense_visual`):
```python
# one-hot per on-cell animal, summed -> a per-class COUNT
on_cell = jnp.all(animal_pos == agent_pos, axis=-1)          # [n_animals] bool
channels = jax.nn.one_hot(params.animal_visual_channel, 8)   # [n_animals, 8]
visual = jnp.sum(jnp.where(on_cell[:, None], channels, 0.0), axis=0)  # [8]
```

**Where it lands.** In the matched-aggression episode-0 read, channel 7 held `2.0` from step 10 on (both rabbits riding), while the predator channel (5) stayed `0.0` pre-contact — the raw evidence that exposed the conditional mechanism the aggregates had hidden.

### A.2 M1 — interrupted-feeding rate

**Plain-English question.** How often does the agent break off eating when a threat is nearby?

**Formula.**
```
M1 = (# steps where the agent was eating last step, a threat is within the threat radius,
      and the agent is NOT eating this step) / (# eating steps with a threat in radius)
```
**Edge case → NaN.** Undefined (NaN) if the agent never eats with a threat in radius.

**Where it lands.** No-predator transfer eval: **M1 = 0.000** — the agent never interrupts feeding for a harmless chaser. (Implementation: `scripts/eval_rollout.py` behaviour-measure block.)

### A.3 M5 — eat-under-threat ratio

**Plain-English question.** Does the agent keep eating while a threat is close, or suppress eating?

**Formula.**
```
M5 = (eat rate when a threat is within the threat radius) / (eat rate when no threat is in radius)
   > 1  -> eats MORE near threats (no suppression)
   < 1  -> suppresses eating near threats
```
**Edge case → NaN.** Undefined if either eat rate is zero.

**Where it lands.** No-predator transfer eval: **M5 = 1.08** (eats slightly *more* near the harmless chaser — no suppression). Under matched smells the May anchor found M5 < 1 for the predator (suppression) — an event-level discrimination signal — but the June controls reframe that as plausibly post-contact.

### A.4 Disengage-on-contact (strike-and-retreat)

**Plain-English question.** Why did the harmless rabbit used to "ride" the agent, and what fixed it?

**Walk-through.** Originally the post-contact pause (which makes an animal back off after touching the agent) only fired on *damaging* contact, so only the predator bounced away; the harmless rabbit kept stepping back onto the agent every tick (it "rode" the agent ~50% of post-contact steps). The disengage feature drains an animal's stamina to zero on **any** contact, forcing it to retreat and recover — applied to both animals, so the strike-and-retreat rhythm is now matched and cannot be a tell for which one is dangerous.

**Code extract** (`src/environment/core.py`, contact block):
```python
# any contact (not just damaging) drains stamina -> animal must back off and recover
new_animal_stamina = jnp.where(
    at_animal & params.animal_disengage_on_contact, 0.0, new_animal_stamina)
```

**Where it lands.** In the cell-08 lethal-contact training the post-contact ride artifact is gone (on-cell fraction ~1% vs the old 50–84%), so the predator↔rabbit behavioural match is tight — and the agent is still class-blind, which is the cleanest form of the study's negative result.

### A.5 Sanity note

These measures are descriptive (no pre-registered numeric gates for the June thread). A measure should be ignored when its denominator is zero (NaN) — e.g. M1/M5 in episodes where the agent never eats under threat. The load-bearing evidence in this study is the **trajectory-level read plus the raw observation vector**, not any single aggregate — which is the methodological lesson the study exists to record.
