---
title: "Experiment-environment design — guiding perspective and directions"
topic: behavior_measures
status: active
created: 2026-06-16
last_updated: 2026-06-16
phase: foundational
aliases: [experiment-env-design-perspective, probe-design-perspective]
---

# Experiment-environment design — guiding perspective and directions

## Purpose (read this first)

We have **one** world where agents learn to survive: a rich, busy environment with
foraging routes, threats, obstacles, and a survival budget. That richness is exactly what
we want for *training* — but it is the wrong place to *measure* what a trained agent has
actually learned. The world blends many situations together, so a single average number
(say, "how far did it stay from the threat?") mixes all those situations and can wash a
real, situation-specific behavior down to near-zero — or invent one that isn't there.

So we split the job into two kinds of measurement:

1. **Training-environment metrics** — what we already read off the live training world
   (survival steps, the existing aggregate signals). Cheap, always available, but prone to
   confounds because the world is complex.
2. **Experimental-environment metrics** — numbers we collect by dropping the **already-trained,
   frozen agent** into small, purpose-built worlds designed to isolate one behavior at a
   time. The agent is **never trained here.** These environments exist only to *interpret*
   the trained agent — to get a clean quantitative read on a specific behavior that the
   training world either confounds or never gives us enough clean data on.

This document sets the **goals and directions** for the second kind. It deliberately holds
no concrete environment specs — those come later. Its only job is to fix the perspective so
every future probe is built for the same reason.

## What an experimental environment is (and is not)

- **It is** an interpretation tool: a simple, controlled world that removes confounds so a
  plain mean-level measure becomes trustworthy again — recovering what a slow hand-read of
  individual episodes would tell us, but at scale across agents and seeds.
- **It is not** a place to train, tune, or reward-shape the agent. No weights change inside
  a probe. A probe that changed the agent would no longer be measuring the trained agent.
- **It is not** a replacement for training-environment metrics. We keep both. The two are
  complementary readings of the same frozen agent.

## The behaviors we want to measure

Starting from basics, the platform targets four core behavior families that any
survival agent must exhibit. Each deserves its own clean probe so its measure is not
contaminated by the others:

- **Foraging** — how the agent seeks, approaches, and consumes resources when nothing
  threatens it.
- **Avoidance** — how the agent detects and keeps distance from / disengages a threat.
- **Recovery** — how the agent restores a depleted internal state (e.g. returning to feed
  or rest) after a disturbance.
- **Managing conflicting needs** — how the agent trades off competing drives when it cannot
  satisfy them at once (e.g. eat vs. flee, feed vs. avoid).

These are the directions. The concrete worlds, parameters, and quantitative measures for
each are defined in later design documents, not here.

## Design principles

1. **Frozen agent, no retraining.** Every probe loads an already-trained checkpoint and
   only observes its behavior.
2. **Isolate one factor.** Each probe removes confounds so a single behavior dominates and
   its mean-level measure becomes honest.
3. **Simple before clever.** Start from the simplest world that exposes the behavior; add
   complexity only when a measure demands it.
4. **Means stay, but measured where they can't be fooled.** We still want scalable
   quantitative numbers — we just collect them in a place built to keep them trustworthy.
5. **Cross-check before trusting.** A mean-level probe measure is only believed once a
   trajectory-level read confirms it tells the same story (the confound that motivated this
   whole platform cuts both ways — aggregates can hide a real effect *and* manufacture a
   fake one).

## Allowed manipulations (degrees of freedom)

A probe is free to depart from the training world wherever the change serves the
measurement. The training world is the *reference*, not a constraint to be preserved. In
particular:

1. **Grid size and shape are free knobs.** A well-trained agent's competence should not
   depend on the arena's dimensions — if it is robust, the size and shape of the world carry
   no meaning for it. So we may shrink, enlarge, or reshape the grid to whatever makes a
   behavior easiest to isolate and read.
2. **Initial internal state is a deliberate dial.** We can set the agent's starting levels of
   satiation, interoceptive / nociceptive state, etc., to *induce* a particular motivation or
   conflict — e.g. start it hungry to force foraging, start it depleted to probe recovery, or
   set two competing needs low at once to force a trade-off. The starting state is part of the
   probe design, not a fixed inheritance from training.
3. **Significant departures are allowed when they meaningfully induce behavior.** Changes that
   diverge substantially from the training world — for example altering a predator's olfactory
   signature or its hunting mechanism — are **not prohibited** if they are what it takes to
   surface the target behavior cleanly. This follows directly from the platform's premise: the
   environment *inducing* behavior is the feature, not a violation. The only obligation is that
   the interpretation account for the change — a measure read under a modified world is a
   statement about behavior *in that world*, and must be reported as such.

## Relation to existing work

- This charter generalizes the earlier, predator-rabbit-specific
  [[behavior_measurement_charter]] — that one was the first concrete instance of this idea
  (isolated test worlds making a mean honest); this one states the broader platform goal
  across all four behavior families.
- The reframe that makes "the environment induces the behavior" a *feature* rather than a
  failure is captured in memory insight `20260616_1514_experimental_env_as_behavior_platform`.

## Open questions and next steps

- **First concrete instalment**: [[experiment_environment_designs_v1]] designs one probe world per behavior family (Foraging worked as the template; Avoidance, Recovery, Conflict), each with its internal-state dial, candidate mean measure, trajectory cross-check, and an explicit anti-geometry-artifact clause.
- For each behavior family, what is the *simplest* world that isolates it? (Future design
  docs, one per family.)
- Which existing training-environment metrics pair naturally with each probe measure?
- How do we validate that a probe measure is trustworthy before relying on it? (The
  trajectory cross-check is the gate; the protocol is TBD.)
