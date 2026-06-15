---
title: "Discrimination-metric search: the apparent predator-vs-rabbit discrimination is a spatial-encounter artifact"
topic: hypervigilance
status: active
created: 2026-06-16
last_updated: 2026-06-16
phase: hypervigilance
aliases: [discrimination-metric-search-results]
---

# Building a discrimination metric — and why no clean one exists (yet)

## Purpose / headline (read this first)

We set out to build a **test environment whose simple averaged measure faithfully reports
whether an agent tells a harmful predator apart from a harmless rabbit** — calibrating it
against a known "discriminator" agent (the Cell C agent, which had shown a large bush-dive
gap) and a known "class-blind" agent (the cell-08 single-predator-rabbit agent). The plan
(per the testbed charter) was: a candidate test world is only trustworthy once its mean-level
measure reproduces the known verdict.

**It did not work — and the reason is the important finding.** Across four test designs, no
world produced a measure that separated the two control agents the way their reputations
required. The decisive result: when we drop the *known class-blind* agent into the *rich,
asymmetric* world where the "discriminator" earned its reputation, the class-blind agent
**reproduces the entire bush-dive gap** (+0.31, essentially identical to the discriminator's
+0.30). And when we control the spatial geometry — either a symmetric common world or a
mirror-counterbalanced arena — **neither** agent shows any discrimination on any measure.

The conclusion is therefore that **the apparent "predator-vs-rabbit discrimination" is a
spatial-encounter artifact, not learned class-recognition.** In the geometry that produced it
(food decoupled to two quadrants, rabbits confined to the other two, predator roaming the
whole grid), the predator simply *encounters the foraging agent in threatening contexts more
often than the spatially-confined rabbits do* — so any agent, even a class-blind one, looks
like it "avoids the predator more." Remove that geometric asymmetry and the effect vanishes.
This mechanistically explains the earlier "class-blind / pain-reactive, not anticipatory"
verdict and resolves the open question (flagged in memory) of whether the event-level gap was
itself spatially mediated. It was.

**Bottom line for the program:** there is no confirmed *genuine* discriminator to calibrate a
metric against. Before a discrimination metric can be built and validated, a genuinely
class-discriminating agent must first be established (e.g. trained with a *learnable distal
class cue*), or the class-blind/artifact conclusion is accepted as the result.

## What was run

All runs are frozen-checkpoint evaluations (no retraining): 200 deterministic episodes each.
Controls: **Cell C** (discriminator, checkpoint 9990005) and **cell-08** (class-blind, final
10M checkpoint 9990029). The discrimination read-out is a contrast (cross-world or within-world)
on the event-level behaviour measures: M2 bush-dive rate, M5 eat-under-threat ratio, plus
closest-approach distance.

| # | Test design | What it isolated | Result |
|---|---|---|---|
| 1 | **Solo worlds** — predator-only vs rabbit-only (+ forage-only) | one animal class per world; cross-world contrast | **FAIL + inverts** — the class-blind agent showed the only suppression; the discriminator looked flat. Isolation destroys the contrast discrimination is made of, and adds a lethality confound (the discriminator dies early in predator-only: 160 vs 443 steps). |
| 2 | **Symmetric common world** (the cell-08 config: 1 predator + 1 rabbit, byte-identical except damage, full-grid, symmetric food) | within-world per-class contrast, geometry matched | **FAIL** — the discriminator's gap collapses from +0.32 to +0.016; both agents flat. Discrimination does not transfer to a geometry-matched world. |
| 3 | **Cell C's native world** (decoupled food, 1 predator + 2 confined rabbits), both agents | agent-driven vs world-driven (the 2×2) | **PIVOTAL** — *both* agents show the big bush-dive gap (Cell C +0.295, cell-08 **+0.307**). The gap is world-driven, not agent-driven. |
| 4 | **Symmetric mirror arena** (static predator vs rabbit at mirror-symmetric halves, predator-left and predator-right runs averaged to cancel side bias) | class signal with encounter geometry fully controlled | **FAIL (clincher)** — neither agent discriminates (Cell C Δdist +0.22 / Δeat −0.012; cell-08 Δdist 0.00 / Δeat −0.016). The iter-3 eat-suppression residual (−0.37) collapses to −0.012. |

## The decisive evidence — the 2×2

Within-world Δ = (measure for predator) − (measure for rabbit), bush-dive rate:

| | **Cell C agent** | **cell-08 agent (known class-blind)** |
|---|---|---|
| **symmetric world** (cell-08 config) | +0.016 | −0.002 |
| **asymmetric world** (Cell C native) | +0.295 | **+0.307** |

Read it by column or by row and the same thing jumps out: the gap tracks the **world**, not the
**agent**. A known class-blind agent produces the full "discrimination" gap in the asymmetric
geometry; the reputed discriminator produces no gap in the symmetric geometry. The eat-suppression
measure tells the same story — Cell C's selective predator-suppression (−0.37) exists only in its
native asymmetric world and disappears (−0.012) the moment geometry is controlled in the arena.

## Why the solo design failed specifically

Discrimination is inherently a **contrast behaviour** — an agent expresses it by treating two
*simultaneously present* classes differently. Splitting them into one-animal worlds removes the
confound but also removes the very contrast, and introduces a lethality confound (the agent's
trajectory in predator-only is truncated by early death). This is a charter-level lesson:
*single-animal isolation is the wrong isolation for a contrast phenomenon.*

## What this means / what's next

- **The calibration controls did their job.** They prevented shipping any of these worlds as a
  "discrimination metric" — each would have reported geometry, not class. This is the whole point
  of calibrate-before-trust.
- **The premise needs revisiting.** The "known discriminator" (Cell C) is not a confirmed
  discriminator once geometry is controlled. The metric-validation program needs a *genuinely*
  class-discriminating agent before any test world can be calibrated.
- **Candidate next moves** (pending the user's call):
  1. Establish a real discriminator: train an agent with a **learnable distal class cue** (a
     sensory channel or memory pressure that *could* support pre-contact class recognition), then
     re-run this calibration. Only if such an agent shows a geometry-robust gap is the phenomenon
     real.
  2. Accept the **class-blind / artifact** conclusion as the result and write it up — it is a clean,
     mechanistically-explained negative that strengthens the June verdict.
- **Caveats.** Single seed per agent; the symmetric tests place the Cell C agent out-of-distribution
  (so its *own* failure to discriminate there could be OOD confusion) — but the airtight leg of the
  argument (a class-blind agent reproducing the gap in Cell C's world) does not depend on that.

## Links / manifest

- Charter governing this: [[behavior_measurement_charter]]
- Pre-registered design + thresholds: [[testbed_solo_validation]]
- Configs: `configs/experiment/hypervigilance/testbed_{predator_solo,rabbit_solo,forage_only,cellC_native_v2,arena_PL,arena_PR}.yaml`
- Eval outputs: `results/eval/testbed/{cellC,cell08}_{predator_solo,rabbit_solo,forage_only,bothmatched,cellCworld,arena_PL,arena_PR}/`
- Controls: Cell C `…round26-C-seed44…/models/9990005`; cell-08 `…disengage_s42/models/9990029`
- Prior framing this refines: [[20260512_1428_sameprop_class_discriminating_defence_event_level]] (its event-level gap is now shown to be geometry-mediated), [[20260609_1747_avoidance_is_post_contact_not_preemptive]]
- Search log (raw iteration trace): `tmp/20260616_002711_discrimination_assay_search.md`
