# Does a neuromodulator change what the agent attends to?

**Shareable page:** https://claude.ai/code/artifact/1351009f-d7f7-4114-a290-f6582bb9a004 (final section)
**Reproduce:** `scripts/analysis/hiding_drivers.py --store-root results/trajectories_nmn` on both
runs, then `scripts/analysis/supplementary/compare_pair.py <unmod.npz> <film.npz>`
**Sibling study:** [[a01_hiding_drivers]] (a different world — see Scope)

## Question

The project's neuromodulation line adds a FiLM modulator: a small network that reads the agent's
internal state and rescales its perceptual features. The hope is that an agent which can gate its
own perception by context should be **better at telling a real threat from a false one** — which
matters here, because the companion study found this environment's agents hide from a harmless
rabbit that happens to smell predator-like, and pay for it in survival.

This document tests that directly, on the cleanest comparison the repository contains.

## The comparison

Two trained runs that differ in **exactly one thing**: whether the agent has a FiLM modulator.

| | unmodulated | FiLM-modulated |
|---|---|---|
| run | `rppo_b04_mc_dp1_n110` | `rppo_nmn_g32_b04_mc_dp1_n113` |
| checkpoint replayed | 45,700,001 | **45,700,001** (exact match) |
| training seed | 42 | 42 |

Environment, sensory, body, perceptual-noise and training configs are **byte-identical**. The only
differences anywhere are the seven keys of the agent's modulation block. Both were replayed over
the **same** 1,000,000 world draws, so every comparison below is paired: same world, different
brain.

Modulation is confirmed live rather than silently dropped — the two agents' behaviour diverges in
2,931 of the first 5,000 paired episodes.

## Headline

| | dwell | survival | killed | starved | food/step |
|---|---|---|---|---|---|
| unmodulated | 14.08% | 174.29 steps | 46.92% | 27.91% | 0.2059 |
| **FiLM** | 14.23% | **177.10 steps** | 47.14% | 27.44% | 0.2170 |

The modulator buys **+2.82 survival steps** (95% CI +2.58 to +3.05), about 1.6%. Hiding is
unchanged (+0.0003, CI spans zero).

### The mean hides almost everything

That +2.82 is a real average, but it is a thin edge on a coin-flip, not a systematic improvement.
Across the million paired worlds:

| | share of worlds | mean effect |
|---|---|---|
| FiLM survives **longer** | 30.7% | +92.7 steps |
| **identical** survival | 40.9% | 0 |
| FiLM survives **shorter** | 28.4% | -90.2 steps |

The two agents produce the *same* survival in four worlds out of ten, and where they differ the
wins and losses are almost mirror images. The distribution of the paired difference has a median
of exactly **0** and a standard deviation of **120 steps** — so the +2.82 mean is **0.023 standard
deviations**. It is measured precisely because a million paired worlds make it so, not because it
is large.

Where the edge does come from is threat:

| world | unmodulated | FiLM | difference |
|---|---|---|---|
| no predator | 404.9 | 404.6 | **-0.29** |
| one predator | 80.3 | 86.5 | **+6.20** |
| two predators | 36.9 | 39.4 | +2.54 |

With nothing hunting it the modulator is very slightly harmful; the whole benefit appears once a
predator is present. That is at least the right shape for a threat-gating mechanism, even if the
magnitude is small.

The outcome itself changes in only 28.2% of worlds. The modulator's clearest directional effect is
on *how* the agent dies: it starves less (27.91% to 27.44%, eating 0.2170 per step against 0.2059)
and is killed slightly more (46.92% to 47.14%).

## Does it improve threat discrimination? The two pairs disagree.

This was the question the modulator was meant to answer. A first pair said no. A second, matched
pair says the opposite, and emphatically. **The effect does not replicate**, so no claim about
discrimination is supportable from this evidence.

The measure is how much each agent leans on the true cue (the predator's scent) relative to the
false one (a rabbit's). Above 1 means it weights the real threat more; below 1 means the false
alarm dominates.

| pair | unmodulated | FiLM | verdict |
|---|---|---|---|
| **b04** | 0.83 | **0.74** | modulator makes it *worse* |
| **b03** | 0.69 | **1.55** | modulator makes it *much better* |

The underlying coefficients (Δpp per SD of scent):

| | b04 unmod | b04 FiLM | b03 unmod | b03 FiLM |
|---|---|---|---|---|
| predator's scent (true) | +3.26 | +2.42 (z -8.0) | +0.44 | **+1.18** (z +10.3) |
| rabbit's scent (false) | +3.95 | +3.25 (z -6.9) | +0.64 | +0.76 (z +1.7, n.s.) |

In the b04 pair the modulator turned both channels down, the true one more. In the b03 pair it
turned the true channel *up* by 170% and left the false one alone — which is exactly the
behaviour the modulator was designed to produce, and it lifts the ratio above 1 for the only time
in this entire study.

Two further observations. The b03 agents barely use scent at all in absolute terms (+0.44 and
+0.64, against +3.26 and +3.95 for b04), so the two pairs did not converge on remotely similar
strategies despite differing only in a baseline-config number. And both *unmodulated* agents sit
below 1, reproducing the false-alarm signature seen in the companion study.

**One run per condition is the whole problem.** Each pair is a single modulated run against a
single unmodulated one, all from seed 42. Two pairs give opposite answers, which is the clearest
possible demonstration that a single pair cannot separate "the modulator does this" from "this
training run did this". Two more matched pairs exist (`b03_gae`, `b04_gae`); four independent
replications would begin to settle it. Until then the discrimination question is **open**.

## What replicates

Three results hold in both pairs, and those are the ones worth carrying forward.

**The modulator barely matters on average, and matters a lot case by case.**

| | which world | the modulator | modulator x world |
|---|---|---|---|
| dwell, b04 / b03 | 74.0% / 74.4% | **0.000% / 0.009%** | 26.0% / 25.6% |
| survival, b04 / b03 | 91.6% / 91.0% | **0.005% / 0.002%** | 8.4% / 9.0% |

Strikingly consistent, and the same pattern as the ten resting-bonus agents.

**A small positive survival effect**: +2.82 steps (b04) and +1.89 (b03). Same direction, similar
size, both tiny against a spread of ~120 steps.

**A shift in how the agent dies**: in both pairs the modulated agent eats more, starves less and
is killed more.

| | b04 unmod → FiLM | b03 unmod → FiLM |
|---|---|---|
| food per step | 0.2059 → 0.2170 | 0.2115 → 0.2220 |
| starved | 27.91% → 27.44% | 34.47% → **30.87%** |
| killed | 46.92% → 47.14% | 39.82% → **43.54%** |

The modulated agent trades safety for food in both pairs. In b03 the trade is large: three and a
half points of starvation converted into nearly four points of predation. That is a coherent
behavioural signature, and it is the most robust thing this comparison produces.

## Factor sensitivities in the b04 pair

Every difference below is precisely measured (paired worlds, 111,211 one-predator one-rabbit
episodes) and highly significant — but as the section above shows, most do not survive
replication. They are recorded for completeness, not as findings.

| factor | unmodulated | FiLM | difference | z |
|---|---|---|---|---|
| food available | -3.89 | -2.39 | **+1.50** | 14.6 |
| predator's attack delay | +1.54 | +0.59 | -0.95 | -9.4 |
| predator's detection range | +10.45 | +11.12 | +0.67 | 6.5 |
| spawn distance to cover | -5.74 | -5.09 | +0.66 | 5.8 |
| starting nutrition | +3.57 | +4.06 | +0.50 | 4.4 |
| predator max stamina | -0.18 | +0.22 | +0.40 | 3.9 |
| starting injury | -1.03 | -1.37 | -0.34 | -3.4 |
| bushes available | +3.80 | +3.50 | -0.30 | -2.9 |
| rocks (null control) | +0.01 | +0.07 | +0.06 | 0.6 |

The largest single change is that the FiLM agent is markedly **less sensitive to how much food is
around**, and slightly **more** sensitive to the predator's detection range — the one cue that
genuinely predicts danger. Both controls (rocks, ambush count) stay at zero in both agents.

Injury-dependent hiding is essentially unchanged: the severe-minus-mild gap is +11.5 points
unmodulated and +12.8 with FiLM. Proximity exposure is identical to two decimal places.

## How much does the modulator matter at all?

Both agent and environment are deterministic given (world, agent), so the outcome table has no
noise term and decomposes exactly.

| | which world | **the modulator** | modulator x world |
|---|---|---|---|
| dwell | 74.0% | **0.000%** | 26.0% |
| survival | 91.6% | **0.005%** | 8.4% |

The modulator's average effect is indistinguishable from zero; what it changes is *which* world
produces *which* behaviour. This reproduces exactly the pattern found across the ten resting-bonus
agents: near-identical on average, materially different case by case.

## Scope and limits

- **One run per condition, both from seed 42.** The `b03` replication was run and *disagrees
  with `b04` on the headline question*, which settles the methodological point: a single pair
  cannot attribute anything to the modulator. Two further matched pairs (`b03_gae`, `b04_gae`)
  exist.
- **The b03 world pins predator attack range to zero**, so that regressor is constant there and
  its coefficient is undefined. Not an error, but the two pairs are not identical worlds.
- **A different world from the companion study.** Here bushes conceal but do **not** block
  predators, and baseline healing is 50x slower. Numbers here are not comparable to
  [[a01_hiding_drivers]]; the two studies stand separately.
- **Replay compatibility.** These runs predate the v3.1 sensor change and need
  `--assume-pre-v31-sensors`. That flag was verified to reproduce the old sensors exactly (max
  observation difference 2.4e-07 against observations the pre-v3.1 code actually recorded).
