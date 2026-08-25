# Does a neuromodulator change what the agent attends to?

**Shareable page:** https://claude.ai/code/artifact/1351009f-d7f7-4114-a290-f6582bb9a004 (final section)
**Reproduce:** `scripts/analysis/hiding_drivers.py --store-root results/trajectories_nmn` on both
runs, then `scripts/analysis/supplementary/compare_pair.py <unmod.npz> <film.npz>`
**Sibling study:** [[a01_hiding_drivers]] (a different world — see Scope)
**Context-dependence tool:** `scripts/analysis/context_dependence.py` — re-run this against new
modulator checkpoints; the metrics are defined to stay comparable across runs and worlds.

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

## Four matched pairs: the modulator has no context-independent effect

All four matched pairs the repository contains were run. Each is one FiLM-modulated agent against
one unmodulated agent that is identical in environment, sensory, body, noise and training config,
trained from the same seed, replayed at a step-matched checkpoint over the same 1,000,000 worlds.

| pair | true/false cue ratio | | survival | food per step |
|---|---|---|---|---|
| | unmod | FiLM | FiLM − unmod | unmod → FiLM |
| **b04_mc** | 0.83 | **0.74** worse | **+2.82** | 0.2059 → 0.2170 |
| **b03_mc** | 0.69 | **1.55** better | **+1.89** | 0.2115 → 0.2220 |
| **b04_gae** | 1.61 | **1.02** worse | **−19.59** | 0.2325 → 0.1793 |
| **b03_gae** | 1.53 | **3.00** better | **−1.77** | 0.2470 → 0.2122 |

Every headline direction splits two-two — and not at random. The splits line up with the setup:

- **Discrimination direction tracks the baseline.** Both `b03` pairs improve, both `b04` pairs
  worsen. The two baselines differ in exactly two settings: whether a predator can **pounce at
  range** (`b04`: reach 2–3 tiles, 50% success) or must make **contact** (`b03`: reach 0). Both
  worlds are genuinely dangerous — `b03` predators still land 3,302 strikes per million steps
  against `b04`'s 4,526 — so this is a difference in *how* threat arrives, not whether it exists.
- **Survival and eating direction track the return estimator.** Both `mc` pairs gain survival and
  eat more; both `gae` pairs lose survival and eat less. The `b04_gae` case is severe: **−19.6
  survival steps** and a 23% drop in eating.

So the modulator does not have an effect that survives a change of setup. Whether it helps or
hurts perception depends on how predators attack; whether it helps or hurts survival depends on
how returns are estimated.

**How much to lean on those alignments.** Each is two pairs against two. For a factor named in
advance, a perfect four-way split falls out by chance about one time in eight, and I chose these
two factors after seeing the results from a field of only two candidates. The alignments are
suggestive of a real interaction and are **not** established. Every run also used seed 42, so a
"setup determines direction" reading cannot be separated from "these particular runs differed".

## What survives all four

One result, and it is the same one the ten resting-bonus agents produced:

| pair | modulator share of variance (dwell / survival) | modulator × world (dwell / survival) |
|---|---|---|
| b04_mc | 0.000% / 0.005% | 26.0% / 8.4% |
| b03_mc | 0.009% / 0.002% | 25.6% / 9.0% |
| b04_gae | 0.010% / 0.265% | 32.4% / 12.3% |
| b03_gae | 0.201% / 0.002% | 30.5% / 10.4% |

**The modulator's average effect never exceeds three tenths of one percent of the variance, while
its interaction with the world is consistently 25–32% (hiding) and 8–12% (survival).** It reliably
changes *which world produces which behaviour* and reliably does almost nothing to the average.

That is a real finding about what this kind of modulator does, and it is the only claim here
supported by four independent instances.

## Context dependence: the measure built for the modulator

Everything above asks whether the modulated agent *behaves* differently. That is the wrong question
for what a modulator does. A FiLM modulator rescales perception **according to internal state**, so
the sharp prediction is that its **response to an external cue should depend more on that state**.

`scripts/analysis/context_dependence.py` measures this directly. For each level of felt pain it
computes the **alarm response** — how much more the agent hides with a predator within two tiles
than without — and then asks how far that response ranges across pain levels. A modulator doing its
job should widen that range.

The tool reports two versions. In the **observed** version the agent's pain is its own doing, so the
pattern describes its situation. In the **randomised** version the environment assigns the wound at
episode start before the agent acts, so the pattern describes its disposition. Only the second is
causally identified, and it is the one quoted here.

| pair | alarm-response range, no modulator | with modulator | difference |
|---|---|---|---|
| b04_mc | 3.2 pp | 4.0 pp | **+0.7** |
| b03_mc | 2.1 pp | 2.3 pp | **+0.2** |
| b04_gae | 4.5 pp | 5.0 pp | **+0.4** |
| b03_gae | 3.0 pp | 4.4 pp | **+1.4** |

**The modulated agent is wider in all four pairs.** This is the only directionally consistent result
across all four in the whole comparison — everything else in this document splits two-two.

Two supporting observations. All eight agents show a **rising** alarm response with randomised
injury (trend +0.83 to +1.92 pp per bin): an agent the world wounded reacts more strongly to a
nearby predator. That is causally identified hypervigilance, present everywhere. And the same
measure applied to **nutrition shows no consistent modulator effect** — two pairs wider, two
narrower. Whatever the modulator is doing, it is doing it with injury.

The effects are small (0.2 to 1.4 pp) and all eight agents share seed 42, so this is not attributable
to the modulator either. It is a **baseline**: the numbers to re-run against when an updated
modulator lands.

## The methodological verdict

An earlier version of this document reported the `b04_mc` result — "the modulator does not improve
threat discrimination" — as a finding. It was withdrawn after `b03_mc` reversed it, and the full
four-pair set now shows every directional claim flipping with the setup.

This is the same design limitation that has recurred across this project: **one run per
condition**. It confounded the ten-arm resting-bonus sweep (one seed per level, with compute node
collinear with the parameter) and it defeats the neuromodulation comparison outright.

The fix is not more conditions. It is **replicate seeds**: three seeds at four settings is
strictly more informative than one seed at ten, because it measures how much two identical runs
differ — the quantity every claim here needs and none of them has.

## Appendix: factor sensitivities in the b04_mc pair

Precisely measured (paired worlds, 111,211 one-predator one-rabbit episodes) and highly
significant — but as above, these do not survive replication. Recorded for completeness, not as
findings.

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

- **One run per condition, every run from seed 42.** All four matched pairs were run; every
  directional claim splits two-two. A matched pair cannot attribute anything to the modulator.
- **The b03 world pins predator attack range to zero**, so that regressor is constant there and
  its coefficient is undefined. Not an error, but the two pairs are not identical worlds.
- **A different world from the companion study.** Here bushes conceal but do **not** block
  predators, and baseline healing is 50x slower. Numbers here are not comparable to
  [[a01_hiding_drivers]]; the two studies stand separately.
- **Replay compatibility.** These runs predate the v3.1 sensor change and need
  `--assume-pre-v31-sensors`. That flag was verified to reproduce the old sensors exactly (max
  observation difference 2.4e-07 against observations the pre-v3.1 code actually recorded).
