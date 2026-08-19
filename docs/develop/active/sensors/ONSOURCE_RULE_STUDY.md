---
title: "The on-source decay rule — is 2.0 a considered value or an inherited one?"
topic: sensors
status: active
created: 2026-08-20
last_updated: 2026-08-20
phase: null
aliases: [onsource-rule-study, decay-2-rule]
---

# The on-source decay rule

**Shareable page:** https://claude.ai/code/artifact/51280ff1-ebf8-4cff-a0ef-1b4344a4b5a8
**Sandbox:** [`onsource_rule_study/`](onsource_rule_study/)
**Decides:** open question 2 of [[DIRECTIONAL_SENSORS_PLAN]] · **Companion:** [[OLFACTORY_EXPANSION_STUDY]]

## Question

The olfactory sensor divides each source's strength by its distance. That breaks at distance zero — a
sampling point sitting exactly on a source — so `sensor.py:14` returns the constant **2.0** instead.

Today the guard fires only when the agent is standing on something. The per-cell expansion samples
five cells instead of one, so it will fire whenever a source is under the agent *or any of its four
neighbours*. Is 2.0 a considered value, or one nobody has looked at since it was written?

## Headline finding

**At the shipped γ = 1.0 the constant is not arbitrary and produces no jump.** `1 / 0.5 = 2.0`, so
the rule is exactly "treat standing on the source as being half a cell away" — the grid's own
resolution limit, and the same principled floor the visual point-spread kernel requires for the same
reason (see [[VISUAL_PSF_MECHANISM_STUDY]] §Implementation notes).

An agent walking onto food reads:

| distance | 5 | 4 | 3 | 2 | 1 | on it |
|---|---|---|---|---|---|---|
| reading | 0.200 | 0.250 | 0.333 | 0.500 | 1.000 | **2.000** |
| change | — | ×1.25 | ×1.33 | ×1.50 | ×2.00 | **×2.00** |

The contact step has the *same* multiplier as the step before it. There is no discontinuity in what
the agent experiences.

**This corrects an earlier claim of mine.** I described this rule as "a factor-two discontinuity that
fires often" and argued for changing it on that basis. That was wrong — I read the special case in the
code and assumed it produced a jump without checking what the agent experiences on an integer grid.
There *is* a genuine discontinuity in the function (for distances between 0.001 and 1 it returns up to
1000, then snaps to 2.0), but entities and cells both sit on integer coordinates, so that region is
unreachable. The discontinuity is real and can never be sampled.

## Where it does become a problem

The correspondence holds at exactly one decay power:

| γ | curve at half a cell | code returns | |
|---|---|---|---|
| 0.5 | 1.414 | 2.000 | 40% too high |
| **1.0 (shipped)** | **2.000** | **2.000** | exact |
| 2.0 | 4.000 | 2.000 | half what it should be |
| 3.0 | 8.000 | 2.000 | a quarter |

At γ = 2 the approach sequence goes ×2.25 → ×4.00 → ×2.00: the smell strengthens ever faster and then
*decelerates* at the moment of contact. That is a real kink — and it appears only when γ is changed,
which [[OLFACTORY_EXPANSION_STUDY]] explicitly contemplates as a way to buy near-field contrast. The
rule is correct today and is a latent trap.

## How often it fires

Measured on the real environment, 102,400 agent-steps, shipped config, counting only sources with a
non-zero chemical signature (rocks and hiding predators ship all-zero signatures, so the 2.0
multiplies to nothing for them):

| olfactory range | cells | % of steps the rule fires |
|---|---|---|
| 0 (today) | 1 | **31.6%** |
| 1 (chosen) | 5 | **73.4%** |
| 2 | 13 | **91.2%** |

Not an edge case in either regime. It already governs about a third of observations and would govern
nearly three quarters after the expansion.

## Options

| | Option | Parity | Correct at γ ≠ 1 | Intent visible in code |
|---|---|---|---|---|
| A | Keep the hard-coded `2.0` | ✅ | ❌ | ❌ |
| **B** | **Half-cell floor, `1 / (0.5^γ)`** | ✅ **bit-identical** | ✅ | ✅ |
| C | One-cell floor, `1 / 1^γ = 1.0` | ❌ | ✅ | ✅ (but erases on/beside distinction) |

Option B's parity claim is verified, not assumed: in float32, `1 / (0.5 ** 1.0)` is bit-identical to
`2.0`, so under the shipped configuration no observation changes and no test changes.

**Recommendation: B.** Not because today's numbers are wrong — they are right — but because they are
right for a reason the code does not record, and that reason silently stops holding if γ moves.

## What this does not settle

Whether contact *should* read twice as strongly as an adjacent cell is a modelling question this study
does not answer; it only shows the current value is internally consistent at the shipped γ. Making
contact more or less salient than the curve implies means deliberately changing the floor distance,
and forfeits parity under any option.

## Reproducing

```bash
cd docs/develop/active/sensors/onsource_rule_study
PY=/home/vncuser/miniconda3/envs/grid_world_pain/bin/python
$PY measure_frequency.py      # real env, 256 envs x 400 steps -> frequency.json
$PY make_onsource_figs.py     # figures 0-4
$PY build_onsource_page.py    # rebuild index.html
```

`index.html` is generated and gitignored; PNGs, scripts and `frequency.json` are tracked.

## Related

- [[DIRECTIONAL_SENSORS_PLAN]] — open question 2 is what this study decides.
- [[OLFACTORY_EXPANSION_STUDY]] — flagged this as open; its γ-sweep question is what makes it a trap.
- [[VISUAL_PSF_MECHANISM_STUDY]] — needs the same half-cell floor for its Gaussian, for the same reason.
- [[09_sensors_and_observation]] §6 — documents `sense_resource`, including the guard at `sensor.py:14`.
