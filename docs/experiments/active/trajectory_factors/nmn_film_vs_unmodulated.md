# Does a neuromodulator change what the agent attends to?

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

A detail worth pausing on: FiLM's *mean* survival is higher, but it survives longer in only
**30.7%** of individual worlds. It loses in most worlds and wins large in a minority. The average
is not the typical case.

## Does it improve threat discrimination? No.

This is the question the modulator was meant to answer, and the answer is negative.

| scent the agent responds to | unmodulated | FiLM | difference | z |
|---|---|---|---|---|
| **predator's** scent (a true signal) | +3.26 | +2.42 | **-0.83** | -8.0 |
| **rabbit's** scent (a false alarm) | +3.95 | +3.25 | **-0.70** | -6.9 |

The modulator does reduce the false alarm — but it reduces the *true* signal **more**. Expressed
as how much the agent leans on the real cue relative to the misleading one:

- unmodulated: 3.26 / 3.95 = **0.83**
- FiLM: 2.42 / 3.25 = **0.74**

Both are below 1, meaning **both agents respond more strongly to the harmless rabbit's scent than
to the predator's** — the same false-alarm signature the companion study found in a different
world. The modulator makes that ratio slightly *worse*, not better. It turns the scent channel
down globally rather than sharpening it.

## What it does change

Every difference below is precisely measured (paired worlds, 111,211 one-predator one-rabbit
episodes) and highly significant, but small.

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

- **One run per condition, both from seed 42.** The differences are measured to three decimal
  places, but a single modulated run against a single unmodulated run cannot separate "the
  modulator does this" from "this particular training trajectory does this". A replication on the
  matched `b03_mc_dp1` pair is running; two further matched pairs (`b03_gae`, `b04_gae`) exist.
- **A different world from the companion study.** Here bushes conceal but do **not** block
  predators, and baseline healing is 50x slower. Numbers here are not comparable to
  [[a01_hiding_drivers]]; the two studies stand separately.
- **Replay compatibility.** These runs predate the v3.1 sensor change and need
  `--assume-pre-v31-sensors`. That flag was verified to reproduce the old sensors exactly (max
  observation difference 2.4e-07 against observations the pre-v3.1 code actually recorded).
