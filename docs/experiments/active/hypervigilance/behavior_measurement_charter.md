---
title: "Behavior-measurement testbeds — why we measure agent behavior in isolated test environments (charter)"
topic: hypervigilance
status: active
created: 2026-06-12
last_updated: 2026-06-12
phase: hypervigilance
aliases: [behavior-measurement-charter, testbed-charter]
---

# Behavior-measurement testbeds — why isolated test environments

## Purpose (read this first)

We train agents in a rich, complex world — many animals, obstacles, foraging routes, a
survival budget. That complexity is good for *training* but bad for *measuring*. When we
asked the central question — **"does the agent tell a dangerous predator apart from a
harmless rabbit?"** — our summary numbers (averages such as mean distance kept, or flee
rate) said **"no, it is class-blind."** But when we read individual episodes step by step,
the agent clearly **did** discriminate: it kept eating when a harmless rabbit approached,
yet **stopped eating and dived into cover** when the predator approached — despite the two
animals having identical smell and identical motion.

The averages were not the wrong *tool*. They were **confounded** by the complex world,
which blends many different situations together and averages the real,
situation-specific behavior away to ~zero. The step-by-step trajectory read recovered the
truth the averages had hidden — but reading trajectories by hand does not scale, and we
need quantitative numbers across many agents and seeds.

**This document is the charter for the fix.** We build a family of **dedicated test
environments**, kept entirely separate from training (no retraining — we drop a *frozen,
already-trained* agent into them). Each one **isolates a single factor and removes the
confounds**, so that a simple **mean-level measure becomes trustworthy again** — it now
reflects what a step-by-step trajectory read would show. We keep using means, because we
need scalable quantitative measures; we just measure them in a place where they cannot be
fooled. The test environment's whole job is **confound removal in service of making the
mean honest** — not replacing the mean.

## The problem, in one concrete example

On the chasing-rabbit agent, every aggregate said *class-blind*: mean predator-vs-rabbit
distance gap ~0, a distance-matched flee rate of **76% vs 76%**, interrupted-feeding
**0.000**. From those we concluded "the agent cannot discriminate" — and defended it for
several analysis turns. Only a **full step-by-step trajectory read** (plus inspecting the
agent's raw 27-number observation vector) overturned it, revealing the predator-specific
stop-eating-and-hide response. The means had averaged a *conditional* behavior across many
states down to nothing. Full write-up: the "MOST IMPORTANT FINDING" callout in
[[20260612_1625_predator_rabbit_discrimination]]; methodology post-mortem
[[20260609_1721_aggregate_stats_hide_conditional_behavior]].

## The principle

1. **Separate training from testing** (the two-tier design of our own EVAAA benchmark —
   Lee et al. 2025: a naturalistic training curriculum plus controlled testbeds that
   isolate one decision process by minimizing irrelevant cues). Train in the complex
   world; *measure* in clean, controlled worlds.
2. **Isolate one factor; remove confounds; the mean becomes an honest proxy.** With the
   confounds gone there is nothing left to mislead the average, so a single number once
   again reflects the behavior.
3. **Trajectory reading is the ground truth; the testbed makes the mean match it.** We do
   not abandon the trajectory read — we use it to *validate* that, in the clean world, the
   mean now agrees with it (see Validation below).

## Why not just read trajectories everywhere

Hand-reading episodes is the most faithful method but it does not scale: it is slow,
partly subjective, and infeasible across many agents, seeds, and conditions. We need a
*scalable quantitative* measure. The testbeds are how we earn the right to trust a number.

## Design rules for any test environment (checklist)

1. **Frozen-checkpoint probe — no retraining.** A test environment is an evaluation world,
   never a training world. We drop an already-trained agent in and roll it out.
2. **Change exactly one factor; match everything else.** Every difference between two test
   worlds must be the thing under study and nothing else.
3. **Readable before contact, or contact impossible.** A clean assay separates
   *anticipatory* (pre-contact) behavior from *post-contact* reaction — the confound that
   defeated earlier reads.
4. **Avoid the known confounds.** Motion / chase, approach-speed / aggression, animal
   *number*, out-of-distribution asymmetry, and pre-contact-vs-post-contact mixing have all
   produced misleading numbers before. Name and neutralize each one in the design.
5. **Same measurement panel across all worlds; report distributions, not just means.**
   Use identical measures everywhere and report quantiles / spread, so a thin or skewed
   bin cannot masquerade as signal.
6. **Always a matched contrast, never a single class.** Discrimination is only defined as
   a difference (e.g. predator-world vs rabbit-world). A single-class number cannot
   separate danger-recognition from general caution.
7. **Cross-check the mean against a `trajectory-story` read** on a few representative
   episodes per world — to guarantee we never again accept a mean that disagrees with the
   trajectory.

## Validation requirement (calibration before trust)

**A new testbed's numbers are not trusted until the testbed reproduces a known result.**
Before using a test environment to judge a new agent, run two control agents through it:

- a known **discriminator** — the Cell C matched-smell agent (bush-dive 0.76 predator vs
  0.44 rabbit; eat-suppression 0.73 vs 1.26), and
- a known **class-blind** agent — the cell-08 single-predator-rabbit agent (bush-dive
  ~0.58 vs ~0.56; eat-suppression ~1.0 vs ~1.0).

The testbed **passes** only if its mean-level measure separates these two the same way the
trajectory read does. If the clean world's mean cannot tell the known discriminator from
the known class-blind agent, the isolation is insufficient and the testbed is reworked
before any new conclusion is drawn from it.

## The first testbeds

| Name | Isolates | What it is |
|---|---|---|
| **predator-solo** (was "E3") | the agent alone with one predator | one-animal ablation of the single-pred-rabbit world; harmful animal only |
| **rabbit-solo** (was "E2") | the agent alone with one rabbit | matched control; harmless animal only |

**Read-out:** the **predator-solo vs rabbit-solo contrast** on the shared panel (survival,
closest-approach distribution, bush/cover use, eat-rate near vs far, contact latency),
cross-checked with a trajectory-story read. The broader menu of candidate testbeds (forced
choice, approach-avoidance conflict, barrier anticipation, graded lethality, etc.) is in
[[predator_rabbit_testbeds]]; this charter governs all of them.

## Registry of test environments

| Testbed | Isolates | Status |
|---|---|---|
| predator-solo | harmful animal alone | designed; configs pending |
| rabbit-solo | harmless animal alone | designed; configs pending |
| _(future testbeds appended here as the program grows)_ | | |

## Links

- Study summary (with the trajectory-vs-mean headline): [[20260612_1625_predator_rabbit_discrimination]]
- Methodology post-mortem (means hid conditional behavior): [[20260609_1721_aggregate_stats_hide_conditional_behavior]]
- Candidate testbed menu: [[predator_rabbit_testbeds]]
- Measure-design memos: [[behavioural_discrimination_assay]], [[threat_discrimination_assays]]
- Trajectory-level analysis tool: `.claude/skills/trajectory-story/SKILL.md`
- Two-tier benchmark paper: `docs/project/references/InteroceptiveAI/sources/` (Lee et al. 2025, EVAAA)
