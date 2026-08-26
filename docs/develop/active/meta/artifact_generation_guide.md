---
title: Artifact generation guide — what went wrong and what to do instead
topic: meta
status: active
created: 2026-08-26
last_updated: 2026-08-26
---

# Artifact generation guide

## Purpose

A shareable HTML artifact was built for the bush-dwell analysis
([[a01_hiding_drivers]]). It took roughly a dozen revisions, and almost every revision was
triggered by the reader catching something rather than by the author noticing it. This document
records those failures so the next artifact starts where this one finished.

It is written as a checklist, not an essay. The **Issues found** table is the part to read before
building anything; the rest explains the entries that need explaining.

---

## 1. Terminology: the single largest source of rework

Roughly half the revisions were terminology. Three rules, in priority order.

### 1.1 Use the project's own config vocabulary

If a thing has a name in `models/config.yaml`, that is its name. Inventing a friendlier one costs
the reader a translation on every occurrence and eventually breaks something.

| invented | correct | how it was caught |
|---|---|---|
| ambush predator | `hiding_predator` | reader |
| share of time hidden | **bush dwell** | reader |
| predator's eyesight | `detection_range` | reader |
| strike delay / reach | `attack_delay` / `attack_range` | reader |
| starting injury / hunger | `start_injury` / `start_nutrition` | reader |
| resting bonus, healing dial | `recovery_accel_rate` | author, after the pattern was pointed out |
| chosen steps | **steps** (the `t=0` row is the initial state, not a step) | reader |

**"Bush dwell" is the model case.** It is shorter, it matches the tooling, and it removed a real
ambiguity the author had created — "hiding" was doing double duty for the behaviour and for the
`hiding_predator` resource.

### 1.2 Never use the project's explanandum as a description

The project exists to explain **pain**. Using "pain" to describe the signal the agent receives
assumes the answer — it treats the agent as *having* the thing under investigation on the strength
of a number in an observation vector. Say **nociception**, which is what is measured and what the
config calls it.

The same trap will apply to any term the project is trying to earn: *suffering*, *hypervigilance*,
*avoidance*, *fear*. Use the measured quantity; reserve the construct for a claim you are prepared
to defend.

Where a philosophical framing bears on the result, state it explicitly rather than smuggling it in.
The artifact gained a short section on whether the signal's content is **descriptive** (a report
about the body) or **imperative** (a command), because the data speak to it: at matched nociception,
an agent that was actually struck behaves differently from one carrying the same signal from a
randomly assigned wound.

### 1.3 Check standard terms are used in the standard sense

Worse than inventing a term is misusing a real one. A `professor-rl` review caught two:

- **"replay"** — in RL this means *experience replay*, a training-time buffer. The document used it
  for fresh rollouts of a frozen policy. Correct term: **evaluation rollouts**.
- **"per standard step"** — meant per standard *deviation*, in a document where "step" means
  environment timestep about two hundred times.

And two that were merely confusing:

- **"gain"** means a multiplicative scale factor; the quantity was an additive difference of
  probabilities — and it sat next to a literal modulator network.
- **"context dependence"** — "context" in RL means the *external* task parameters, so this read as
  the opposite of the intended meaning. Now **internal-state dependence**.

**Get a domain agent to review terminology before publishing.** `professor-rl` for RL and analysis
terms, `professor-pain-modeling` for construct validity on anything pain-adjacent.

---

## 2. Figures

### 2.1 Both axes, named in words, on every chart

Not a tick-label. A title, saying what the quantity is and its unit. The reader's exact words:
*"I don't know which are left axis and right axis."*

Where a chart has two y-axes, colour is not enough on its own: draw each axis line in its series'
colour and say so in the legend ("green line, read against the left axis").

### 2.2 Percentage or percentage point — always say which

The most common ambiguity. "+16.0" could be a percentage, a percentage point, a count or a ratio.

### 2.3 State when panels are not comparable

A small-multiples grid where each panel is scaled to its own range invites the reader to compare
heights across panels, which is wrong. Say "compare shapes, not heights" on the figure itself.

### 2.4 Measure label lengths against the space they have

Three rotated axis titles overflowed their plot height, one by more than double. Estimate
`chars × 0.55 × font-size` against the available span and keep under ~90%.

### 2.5 An internal analysis wants more figures than an introduction

The reader asked for this directly. A results document is not a landing page; it should show the
data, not summarise it.

---

## 3. Technical traps

### 3.1 A CSS class beats a `fill` attribute in SVG

`.ax { fill: var(--muted) }` silently overrode every colour set via a `fill` attribute. The axis
numbers *were* being coloured in the source and rendered grey. **Set colour as an inline `style`**,
which wins over the class. This also affected white text inside dark bars.

### 3.2 Validate the chart script with a parser

There is no browser here, and a single syntax error blanks every figure after it. Installing
`esprima` and parsing the script caught a real error that would have destroyed a figure.

```python
import esprima, re
esprima.parseScript(re.sub(r'^\s*<script>|</script>\s*$', '', src, flags=re.S).strip())
```

### 3.3 Check every render target is written to

Two figures rendered blank in this artifact and neither was noticed by the author:

- **Renaming a display label also renamed a data lookup key.** Two dose-response panels vanished.
- **A chart's code was deleted during an unrelated rewrite.** Figure 5 showed headers and no rows
  for several revisions until the reader asked what had happened to it.

Both are caught by one structural check: collect every `id` the document renders into, collect
every id the script writes to, and assert the sets match.

```python
targets = set(re.findall(r'<(?:div|tbody) id="([a-z0-9]+)"', body))
filled  = set(re.findall(r'put\("([a-z0-9]+)"', js)) | ...
assert not (targets - filled), sorted(targets - filled)
```

Prefer a loud placeholder — `[no data for X]` — over a silent `return`.

---

## 4. Structure

Each analysis carries two blocks, and both were added only after the reader asked:

- **Before the figure — "Why this analysis".** What question prompted it, and what would count as an
  informative answer. This matters more than it sounds: most figures in a real analysis exist
  *because an earlier one raised a doubt*, and that thread is invisible to anyone who was not there.
- **After the figure — "How this was computed".** Subset and n, binning, what was conditioned on,
  and **any known weakness in that specific number**. State the biases that make the result weaker,
  not only the ones that make it robust.

Also worth carrying forward:

- A short **"How to read this document"** panel explaining the block types.
- A **vocabulary panel** mapping each term to its config key.
- A **corrections** section listing claims withdrawn or changed during the work, so a reader who
  remembers an older number can see why it moved. This artifact needed four.

---

## 5. Reproducibility

One figure, one script — `scripts/analysis/figures/`. The previous arrangement, a few scripts each
computing several unrelated things, is what allowed a figure's code to be deleted without anything
noticing. One script per figure makes a missing figure a missing *file*, and the merge step refuses
to build a partial dataset.

Rewriting the scripts this way also surfaced **undocumented drift**: one figure's number moved from
20.7% to 18.9% because the original conditioned on `t>=2` and the rewrite on `t>=1`. Neither is
wrong; the difference was invisible until the choice had to be written down.

---

## 6. Issues found in this artifact

| # | Issue | Found by | Fix |
|---|---|---|---|
| 1 | Invented terms instead of config names | reader | §1.1 |
| 2 | "pain" used for the measured signal | reader | §1.2 |
| 3 | "replay" and "per standard step" misused | `professor-rl` | §1.3 |
| 4 | "gain" and "context" misleading | `professor-rl` | §1.3 |
| 5 | Two y-axes indistinguishable | reader | §2.1, §3.1 |
| 6 | Axis titles overlapping panel titles | reader | §2.4 |
| 7 | Rotated titles overflowing their plots | author, after #6 | §2.4 |
| 8 | Per-panel scaling not stated | author, after #6 | §2.3 |
| 9 | Two panels blank after a rename | author, by checking | §3.3 |
| 10 | Figure 5 blank — code deleted | reader | §3.3 |
| 11 | JS syntax error would have blanked a figure | author, via parser | §3.2 |
| 12 | No motivation per analysis | reader | §4 |
| 13 | No method detail per analysis | reader | §4 |
| 14 | Too few figures for an internal document | reader | §2.5 |
| 15 | Untested factors omitted from the ranking | reader | audit the data, not the config |
| 16 | "mistake" asserted without evidence | reader | §7 |
| 17 | Figure 5 still blank in the COMMITTED file — source fixed, HTML never rebuilt | readability review | §3.3, and rebuild before every publish |
| 18 | Agent and episode counts wrong (14 / 14M vs 18 / 18M) | readability review | §9 |
| 19 | Withdrawn-claim count disagreed with itself (four vs two) | readability review | §9 |
| 20 | A caption's comparator ("twenty-five points") appeared in no figure | readability review | §9 |
| 21 | Figure 8's caption described a quantity the figure does not plot | readability review | §9 |
| 22 | Pair names `b03`/`b04` used on four figures, never decoded | readability review | §9 |
| 23 | Flagship figure's axis omitted its own unit ("per 1 SD") | readability review | §2.2 |
| 24 | Nociception's numeric scale never stated | readability review | §9 |
| 25 | Outcome variable had five different names across axes | readability review | §1.1 |
| 26 | Statistical terms undefined (quasi-binomial, overdispersion, collider, 1 SD, pp) | readability review | §9 |
| 27 | Action set, episode cap and hazards never stated | readability review | §9 |
| 28 | No conclusion — the document's three questions never answered together | readability review | §9 |
| 29 | Retracted values still shipping inside the data blob | readability review | §9 |

---

## 7. Claims

The reader asked what "the mistake it makes" meant, and the honest answer was that the word was not
earned. The agent responded proportionally to genuine evidence — an animal smelling that
predator-like really is a predator 83% of the time — and 57.4% of the two odour distributions
overlap, so no policy can separate them on smell alone. What was measured is the cost of
irreducible ambiguity, not an error.

Three claims had to be kept apart, and the document had been sliding between them:

- **"cannot discriminate"** — false; it discriminates well at close range
- **"makes a mistake"** — unearned; the response tracks the evidence
- **"pays a cost for irreducible ambiguity"** — what is actually shown

Before writing an interpretive noun — mistake, failure, hypervigilance, avoidance — ask what result
would have made it false. If nothing would, it is a label rather than a finding.

---

## 9. What only a fresh reader finds

The failures above were mostly mechanical. A separate review, by an agent told to read as a
colleague who knows the project but was not present, found a different class — and found more of
them than every other check combined. They divide into three kinds.

**Things that were true once.** Counts drift as work continues. This document said fourteen agents
and fourteen million rollouts; by the end there were eighteen of each. It promised four withdrawn
claims in one place and two in another. A caption cited "twenty-five points" as its comparator and
no figure contained that number. **Recompute every headline number from the data at publish time**
rather than carrying it forward by hand.

**Things the author cannot un-know.** The four pair names `b03_mc`, `b04_mc`, `b03_gae`,
`b04_gae` appeared as axis labels on four figures and were never decoded anywhere. The nociception
axis was binned three different ways with no statement of what scale it was on. The action set,
the episode cap and the list of things that can injure the agent were never given. Every one of
these was obvious to the author and invisible to the reader.

**Claims with nowhere to land.** The document had no conclusion. It opened by promising three
answers and never put them in one place — the reader had to assemble them from eleven sections.

The cheapest fix for all three: **a setup panel early** (grid, actions, cap, hazards, what "near"
means, per-step vs per-episode) and **a findings box late** (the opening questions, answered).
Neither existed until a reader asked for them.

One further habit worth keeping: the reviewer checked each figure *with the prose covered up*. A
figure whose units are only in its caption fails that test, and several here did.

## 8. Pre-publication checklist

- [ ] Every term matches its config name, or is defined at first use
- [ ] The project's explanandum is not used as a description
- [ ] A domain agent has reviewed the terminology
- [ ] Every chart has both axes titled in words, with units
- [ ] Percentage vs percentage point is unambiguous everywhere
- [ ] Non-comparable panels say so
- [ ] Rotated labels fit their span
- [ ] The chart script parses
- [ ] Every render target is written to; no silent skips
- [ ] Every analysis has motivation before and method after
- [ ] Known weaknesses are stated, not only strengths
- [ ] Corrections made during the work are recorded
- [ ] One script per figure, and the merge step fails loudly
- [ ] Every interpretive claim has a result that would have falsified it
- [ ] The HTML has been **rebuilt** since the last source edit, and republished
- [ ] Every headline count recomputed from the data, not carried forward
- [ ] Internal counts agree with each other (agents, episodes, corrections)
- [ ] Every abbreviation and code name used on a chart is decoded somewhere
- [ ] Setup panel present: grid, action set, episode cap, hazards, what "near" means
- [ ] Findings box present: the opening questions, answered together
- [ ] Each figure read once with the prose covered — do its own labels carry it?
- [ ] No retracted values left in the shipped data
