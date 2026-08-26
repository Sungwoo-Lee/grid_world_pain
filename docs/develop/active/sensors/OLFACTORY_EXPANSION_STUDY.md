---
title: "Directional olfaction — per-cell sampling mechanism study (signal budget, diamond size, superposition)"
topic: sensors
status: active
created: 2026-08-19
last_updated: 2026-08-19
phase: null
aliases: [olfactory-expansion-study, per-cell-olfaction]
---

# Directional olfaction — per-cell sampling mechanism study

**Shareable page:** https://claude.ai/code/artifact/bf93635c-66c5-4013-84c6-1e438c4b663e
**Sandbox:** [`olfactory_expansion_study/`](olfactory_expansion_study/) — pure numpy, independent of `src/`.
**Companion:** [[VISUAL_PSF_MECHANISM_STUDY]] — the same treatment for the visual sensor.

## Question

Olfaction returns 5 numbers: the summed chemical signature of everything in range, evaluated at the
agent's own cell. It says how much of each smell is present and nothing about where it comes from — an
agent standing in a food gradient cannot tell which way is uphill without moving and comparing across
time.

The agreed change is the smallest one that fixes this: evaluate the same function at every cell of a
Manhattan diamond instead of at one point. The field is unchanged; only the number of sampling points
changes. The two questions this study answers are **how much directional signal is actually there**
(a difference between two nearly equal numbers can be arithmetically real and practically useless)
and **what it costs** in observation dimensions.

## Headline finding

**At the shipped γ = 1.0 and the configured olfactory σ = 0.2, the directional signal falls below the
noise floor between two and three cells.**

The level at distance d goes as `d^-γ`, but the quantity that carries direction is the *difference*
between adjacent cells, which is its derivative and goes as `γ·d^-(γ+1)` — one power of d steeper.
Direction therefore always fades faster than presence:

| source distance | level at agent | adjacent-cell difference | difference ÷ σ |
|---|---|---|---|
| 2 | 0.500 | 0.500 | **2.50** |
| 3 | 0.333 | 0.167 | **0.83** |
| 4 | 0.250 | 0.083 | **0.42** |
| 5 | 0.200 | 0.050 | **0.25** |
| 8 | 0.125 | 0.018 | **0.09** |

Two qualifications, stated precisely rather than hand-waved:

- Perceptual noise is **off by default** (`perceptual_noise.enabled: false`), so this binds only in
  the noise experiments — which is to say, precisely the experiments this project exists to run.
- The policy is recurrent and can integrate across steps. Averaging K observations buys √K, which is
  one to two cells of range for realistic K, not an order of magnitude.

Raising γ does not rescue it: contrast improves near the agent and reach shrinks at the same time, so
the usable band shifts inward rather than growing (Fig 5). **On a 10×10 grid this is a close-range
sense.** That is an argument for pairing it with vision, not for tuning γ harder.

## The diamond-size result (Fig 3)

Measured with a planar least-squares fit over all diamond cells — a readable proxy for the
information present in the readout, not a claim about what the network does:

| | range 1 (5 cells) | range 2 (13 cells) | range 3 (25 cells) |
|---|---|---|---|
| median bearing error at d=4, σ=0.2 | 59° | 30° | 14° |
| median bearing error at d=6, noiseless | 0.69° | 1.20° | 1.83° |

**A larger diamond helps substantially once noise is on**, because it supplies redundant samples to
average; range 1 offers only four usable neighbours and no redundancy at all. With no noise the
ordering reverses beyond about three cells, since a plane fitted over a wider patch of a curved field
carries more bias. Textbook bias–variance: the extra cells earn their keep exactly to the extent that
noise is switched on.

**This corrected an earlier claim of mine.** A first pass estimated the bearing from the four
immediate neighbours only, which is insensitive to diamond size by construction, and concluded range 2
bought nothing. That was a property of the estimator, not of the sensor.

## What it costs

| `olfactory_grid_range` | cells | olfaction dims | total observation |
|---|---|---|---|
| 0 (today) | 1 | 5 | 27 |
| 1 | 5 | 25 | 47 |
| 2 | 13 | 65 | 87 |
| 3 | 25 | 125 | 147 |

Range 2 more than triples the observation and leaves three quarters of it as olfaction. Range 3 would
make the chemical sense larger than everything else in the agent's world combined. Whatever Fig 3 says
about accuracy, this is the counterweight.

## Cost, and why there is nothing to optimise

Benchmarked against the real environment, `num_envs: 128`, via
[`olfactory_expansion_study/bench_olfaction.py`](olfactory_expansion_study/bench_olfaction.py). Three
candidate implementations, checked for **both** speed and bit-exactness against today's single-point
sensor:

| implementation | bit-identical at range 0 | range 0 | range 1 | range 2 | range 3 |
|---|---|---|---|---|---|
| today, single point | — (baseline) | 11.0 µs | — | — | — |
| **A — vmap the untouched `sense_resource`** (plan's choice) | ✅ | 10.0 µs | 11.0 µs | 10.7 µs | 12.9 µs |
| B — explicit broadcast, pools separate | ✅ | 10.0 µs | 10.3 µs | 10.6 µs | 13.1 µs |
| C — one concatenated `[C,E] @ [E,V]` matmul | ❌ **max diff 2.4e-07** | 8.5 µs | 11.7 µs | 10.9 µs | 14.2 µs |

`env.step` on the same machine: **398 µs**.

**The diamond is free.** Expanding from one sampling point to five costs **0.00% of `env.step`** — the
marginal cost is inside the measurement noise. Even 25 cells costs +0.47%. As with the visual kernel,
the delta does not scale with cell count although the arithmetic scales 25×, which is the signature of
a launch-bound rather than FLOP-bound operation.

**The obvious optimisation is a trap.** Form C — the single concatenated matmul that mirrors the
visual sensor and looks like the "proper JAX way" — is **not faster at any range that matters**, and
it **breaks bit-exactness**: concatenating the three entity pools changes the floating-point
accumulation order, giving a max absolute difference of 2.4e-07 against today's value. That would
forfeit range-0 byte-parity, the property the entire plan rests on, in exchange for nothing.

Forms A and B are both bit-identical and indistinguishable in speed. The plan keeps A because it
reuses `sense_resource` verbatim, so there is no second copy of the formula to drift.

Micro-optimisations that were considered and rejected: `rsqrt` for the γ=1 case, working in squared
distance to avoid the square root, and fusing the olfactory and visual weight matrices into one pass.
Each changes the arithmetic form, each therefore risks the last bit, and each would save a fraction of
an operation that is already unmeasurable. There is no speed argument available here — only a parity
one.

## Superposition (Fig 4)

Contributions add before the agent ever sees them, so the sensor returns the gradient of the **sum** —
one direction, pointing at the intensity-weighted centroid. With two food sources 90° apart the
readout points at 45°, where there is no food at all. This is inherent to a summed chemical field
rather than a flaw in the expansion, but per-cell sampling is what turns it into a direction the agent
can act on, and act on wrongly. Worth knowing before interpreting any approach-behaviour metric.

## Decisions

### Settled

| Item | Decision |
|---|---|
| Sampling | For each diamond cell, run the existing `sense_resource` with that cell as the sampling point — distances and the `sensor_radius` cutoff both measured from the cell, not the agent. |
| Parity | `olfactory_grid_range: 0` reproduces today's observation byte-for-byte. The centre cell is the current computation unchanged, so parity is structural rather than something to test for. |
| Scope | Olfaction gets the diamond and nothing else — no mask key, no added noise. It already decays with distance, and an entity is already made unsmellable by an all-zero signature. |

### Open

| Item | Options |
|---|---|
| Sensor range | Fig 3 argues for 2 over 1 whenever noise is on; Fig 5 argues against going further; the dimension table argues loudly against 3. |
| Decay power γ | Leave at 1.0, or raise to buy near-field contrast at the cost of reach. No value serves the whole grid. |
| The on-source rule | `sense_resource` returns decay 2.0 when the sampling point sits exactly on a source. Today only the agent's cell can trigger it; with per-cell sampling **any** diamond cell can, so a factor-two discontinuity that fires rarely today would start firing often. Keep, or make it continuous? |
| Dimension budget | Whether 87 dimensions is acceptable, given olfaction would then dominate the observation. |

## Relationship to the visual sensor

After this change the two exteroceptive senses are **the same operator with different kernels**: a
weight per (cell, source) pair, matmul'd against the sources' property vectors. Olfaction's kernel is
global and heavy-tailed (`1/d^γ`, summed, unnormalised); vision's — as proposed in
[[VISUAL_PSF_MECHANISM_STUDY]] — is local, Gaussian, and mass-normalised. Everything else is shared.

Worth knowing before implementing either, since one piece of machinery can serve both; and worth
saying in a paper, since the difference between the two senses reduces to a choice of kernel rather
than a difference of architecture.

## Reproducing

```bash
cd docs/develop/active/sensors/olfactory_expansion_study
PY=/home/vncuser/miniconda3/envs/grid_world_pain/bin/python
$PY make_olf_figs.py    # figures 0-5
$PY build_olf_page.py   # rebuild index.html
```

`index.html` is generated (base64-embedded PNGs) and gitignored; the PNGs and scripts are tracked.
Every number quoted in the page's prose is computed at build time from the same code the figures use.

## Related

- [[ONSOURCE_RULE_STUDY]] — resolves this study's open question about the on-source 2.0 decay
  rule. Headline: at the shipped γ=1 the constant is exactly a half-cell floor and produces no jump;
  it becomes wrong only if γ is swept, which this study proposes.
- [[VISUAL_PSF_MECHANISM_STUDY]] — companion study, and the source of the shared-kernel framing.
- [[09_sensors_and_observation]] — canonical sensor reference. **§6 is stale**: it records
  `decay_power` default 2.0 where `default.yaml` ships 1.0, and olfactory signatures that no longer
  match the config. Every number in this study uses the config values, not the doc's.
- [[10_perceptual_noise]] — the σ = 0.2 olfaction setting this study measures against.
- [[OLFACTORY_PROPERTY_VARIANCE]] — per-episode chemical-signature sampling.
