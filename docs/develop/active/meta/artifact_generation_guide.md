---
title: Artifact generation guide — what went wrong and what to do instead
topic: meta
status: active
created: 2026-08-26
last_updated: 2026-09-01
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

## 11. Three standing requirements for every results artifact (2026-09-01)

These were asked for directly and are now **enforced at build time** by `build_artifact.py`, which
refuses to publish a page that violates any of them. They are requirements, not preferences.

### 11a. Every figure caption states both axes, in words

The caption of every figure carries an `**Axes.**` sentence naming what is on the x-axis and what is
on the y-axis, with units, for every panel. **Repetition across figures is wanted, not avoided** — a
reader who lands on figure 11 should not have to scroll back to figure 4 to learn the convention. If
two panels share a scale, say so; if an axis does not start at zero, say so; if an axis is a
difference rather than a rate, say that too.

*Enforcement:* the build fails if a `<figcaption>` lacks an `<b>Axes.</b>` sentence.

### 11b. Every figure declares how much data it used

Several figures filter — a third of episodes contain no predator, one regression needs exactly one
predator and one rabbit — and a number cannot be judged without the denominator it came from. Every
figure carries a **used / available / percentage** breakdown with one row per distinct subset, and a
plain-English reason for each subset.

The counts are **emitted by the figure script**, never typed: each script calls
`L.record_samples(stem, rows)` with `{what, used, total, note}`, and the percentage is derived. A
figure that records nothing is a hard build error, exactly like a figure with no generating script.
Written by hand, these would drift the first time a filter changed — which is the F15 lesson applied
before it bites.

*Real numbers this surfaced on the sensor-ladder page:* the predator-distance panels use 38.9% of
step rows; the odour regression uses 11.1% of episodes; Figure 15's correlation rests on **14 points**.
None of those were visible before.

### 11c. "How it is computed" is written for a colleague who was not in the room

The method block is the part a reader forwards to a collaborator, so it is written for someone who
knows the science but not this analysis. Target **150–250 words**, covering: what quantity is
computed, in plain terms; the exact subset and why; the conventions that would otherwise surprise
(the `t=0` row is not a step; predictors come from the previous row); any statistical machinery named
*and glossed in one clause* (what quasi-binomial means, what overdispersion is, what "per standard
deviation" means and what it is not); and what the figure does **not** show. Before this rule the
blocks averaged 50 words and five figures had none at all; they now average about 185.

*Enforcement:* the build fails if a figure has no "How it is computed" block.

### A trap found while implementing these

Inserting a block into each of fifteen repeated `<figure>` elements with a `.*?` regex under
`re.DOTALL` **crosses element boundaries**: the pattern for figure 3, finding no block of its own,
matched forward into figure 4 and overwrote that one instead. Figure 4's own pass then rewrote it,
so the net effect was one figure silently left without a method block and no error anywhere. When
editing one of N repeated structures, slice out that element first and operate inside it — never let
the pattern span them. The build check now catches this class regardless.

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
- [ ] Every caption carries an **Axes.** sentence naming x and y, with units (11a)
- [ ] Every figure declares used / available / percentage, emitted by its script (11b)
- [ ] Every 'How it is computed' block is 150-250 words and glosses its jargon (11c)

---

## 10. Issues found in the sensor-ladder artifact (2026-08-30)

A second full artifact, fifteen figures over fourteen agents, reviewed by a fresh-reader agent and
an adversarial analysis reviewer. The presentation lessons from §1&ndash;§9 mostly held; these are
the ones that were new, plus three **analysis** errors that no amount of figure polish would have
caught. That is the headline lesson of this round: the reviewers who found the worst problems were
not looking at the figures at all.

### 10a. Analysis traps that produce a plausible wrong number

| # | What happened | The general rule |
|---|---|---|
| 30 | A third of episodes contained no predator. `numpy.digitize` files every `NaN` into the **top** bin and `numpy.clip` files every `inf` into the **farthest** bin, so those episodes silently became the comparison group and manufactured a large fake effect | **Any bin edge is also a silent dumping ground for missing values.** After binning, print the count per bin and ask whether the extremes are suspiciously fat. A bin holding 3&times; its neighbours is the tell |
| 31 | "Each row is a change of exactly one setting" was true of eleven rows and false of one, inflating a headline number fourfold. The doc even claimed the pairing was "checked against the configs" — no such check existed | **A claim about the data that is stated in prose but not asserted in code will eventually stop being true.** If a figure's caption makes a structural promise, write the assertion that enforces it and let the figure refuse to draw |
| 32 | Two settings became "different" only because one was inert — turning blur off leaves its scale in the config, reading as an extra difference | **Collapse conditional settings before diffing them.** A key that only has meaning when a flag is on is not an independent axis |
| 33 | A sign reversal was explained by a mechanism that, worked through, predicts the opposite sign. It sounded plausible and survived a first draft | **Check that your stated mechanism predicts the sign you observed.** Write the causal chain out and follow the arrows; "dilution" and "weighting" arguments are especially easy to get backwards |
| 34 | The effect changed sign with the measurement window, which looked like window-shopping. It was not — the cause itself decays — but the report had no way to show that | **When the cause is transient, measure and plot its lifetime.** An effect that tracks its cause through time turns the fragility objection into a dose-response confirmation. Sweep the window and show the curve rather than defending one number |
| 35 | "Thirteen of fourteen arms agree" was written as if fourteen independent trials. They shared a seed, shared bit-identical evaluation worlds, and one subgroup shared an observation width | **Count of agreeing conditions is not a count of independent confirmations.** Say what the units actually share. Never multiply them together as if they were coin flips |
| 36 | A grouping rule that separated the arms perfectly was written after seeing which arms separated | **Say when a rule is post-hoc.** It costs one clause and it is the difference between a description and a claim |
| 37 | The "control" channel in a comparison was itself suppressed by a known property of how the variable was defined, inflating the contrast it was meant to anchor | **Interrogate the control as hard as the treatment.** A control that is not neutral is worse than none, because it looks like rigour |

### 10b. Presentation issues that were new this round

| # | What happened | The general rule |
|---|---|---|
| 38 | Seven figures carried a y-axis label saying "poorest senses at the bottom" while plotting the poorest at the top — `np.arange(n)[::-1]` reverses the axis, not the meaning | **Read every axis label against the rendered image, not against the intent.** Orientation claims are the easiest thing to get backwards and the hardest to notice |
| 39 | Red and blue carried five different meanings across the set, twice within a single image | **Fix the colour&rarr;meaning map once, in the shared style module, with a comment saying why.** A reader who learns a colour on figure 3 must not be punished for carrying it to figure 8 |
| 40 | Three multi-panel figures gave each panel its own scale. In the worst case the panel showing the *wrong* answer looked as strong as the right one — in a figure whose whole point was that they differ | **Panels a reader is asked to compare must share a scale.** If one then looks flat, that is the finding, not a rendering problem |
| 41 | Two figures drew 14 and 28 lines in one continuous colour ramp; adjacent arms in the ramp were exactly the arms the text asked the reader to tell apart | **More than about six series needs a grouping, not more colours.** Colour by the distinction the argument turns on, draw the rest thin, and name the ones the prose discusses directly on the line |
| 42 | An axis label was clipped mid-word in the source PNG and shipped | **Open the rendered file, at full size, every time.** Matplotlib clips silently |
| 43 | Figure numbers ran 1,2,3,4,5,12,13,6,7… because two figures were introduced out of order | **Name the script for its figure number and keep them in reading order.** Renumbering is a scripted rename; a reader following numbers that jump is lost |
| 44 | The colour bar was the only place a normalisation was stated, in rotated 7pt text on the right edge | **A units statement that exists only inside a figure has not been made** |
| 45 | A correlation across 14 arm-level points sat next to a caption saying "over that arm's 300,000 episodes" | **Say what n is for every statistic, in the caption.** Especially when a big number is nearby for a different reason |
| 46 | Not every analysis used all the episodes — one used 11% — and no figure said so | **State the denominator per analysis, once, in a table.** Exclusions are not footnotes |

### 10c. What worked, and is worth repeating

- **Generate every table and number; transcribe none.** `make_report_tables.py` emits the report's
  tables, `build_artifact.py` substitutes them into the page, and a token naming a missing figure or
  table is a hard error. No blank panel could ship this time, which was issue #9 last round.
- **One script per figure, named for its figure number**, each stating its own question, method and
  known limitations in its docstring. When a number moved, exactly one file had to change.
- **A verification pass that checks the prose against the data.** Every quoted number was re-derived
  and compared to what the document says. It found nothing this round only because the tables were
  generated — which is the point.
- **Two reviewers with different objects.** The fresh reader found the jargon, the overlaps and the
  misleading scales; the adversarial reviewer found the confounded pairing and the backwards
  mechanism. Neither would have found the other's list. Budget for both.
- **Write the corrections into the artifact.** Each of the three analysis errors is now a short note
  in the report saying what the wrong number was and why. A reader who reproduces the old result
  learns why it differs instead of doubting the new one.
