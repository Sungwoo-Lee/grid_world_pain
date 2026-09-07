---
title: What each sense buys - the fourteen-arm sensor ladder
topic: sensor_ladder
status: active
created: 2026-08-30
last_updated: 2026-09-03
---

# What each sense buys

**Shareable page:** https://claude.ai/code/artifact/3d191a81-2aec-4f97-a3e8-993568936b5b
**Republish:** rebuild with `python scripts/analysis/ladder/build_artifact.py`, then publish
`docs/experiments/active/sensor_ladder/sensor_ladder.html` to the URL above. From a session that
did not publish it, read that URL first and pass it as `url` &mdash; publishing without it makes a
second, separate artifact instead of a new version of this one.


> **A word changed, the quantity did not.** This study used to call the measure **bush dwell**; it is
> now **bush hiding** — same quantity, same numbers, clearer name, changed 2026-09-07 across the
> project. **The figures still show the old word, and so do the data-accounting lines beneath them**,
> because both were produced by a run predating the rename and re-running fifteen figures to change
> one word buys nothing. A y-axis reading "bush dwell" beside a caption reading "bush hiding" is the
> same measure, not a stale number.


## Question

An agent lives in a 10&times;10 grid world. It walks one square at a time, in any of eight
directions. There is food it has to eat or it starves, predators that hunt and kill it, harmless
rabbits, rocks, and bushes it can stand in to hide. Its job is to stay alive as long as possible,
and that is how we score it: **survival steps**, never accumulated reward.

We trained **fourteen** of these agents with the same reinforcement-learning algorithm, on the same
world, from the same random starting point. They differ in exactly one thing: **what they are
allowed to sense**. One can only smell, and cannot even tell which direction the smell is coming
from. One sees a sharp picture of the four squares around it. One sees a blurred picture but can
tell a predator from a rabbit at a glance. One sees perfectly except that rocks and bushes get in
the way. Throughout this document each of the fourteen is called an **arm**, and the set of them a
**ladder**, because most of them sit one single setting away from another one.

Then every agent was turned loose in the **same 1,000,000 test worlds** &mdash; same food, same
predators, same starting injuries. Each agent then behaves differently and so lives out a different
episode, but the world it was handed was identical. So when two agents differ, the difference is the
sensory change and not the luck of the draw.

**Three things we found.** A *direction* for the sense of smell buys more than sharper eyesight
does. What matters about sight is whether it tells you *what* you are looking at, not how sharp it
is. And hiding turns out to be a cost the senses help the agent avoid paying, not a benefit they
help it collect &mdash; the agents that hide most are the ones that die soonest.

We also asked what an **injury** does to behaviour. This world lets us ask that cleanly: at the
start of every episode the agent is handed an injury level drawn at random from 0 to 100 that it did
nothing to earn, and it has no sensor that reads that number. The only route from a wound to
behaviour is an internal channel reporting a smoothed running trace of how injured the body is. So
any behaviour that tracks the assigned wound is *caused* by it.

### The seven findings

Each is tagged with how much weight it can bear, for the reason given in the caveat below.

1. **The most valuable single thing we gave the agent was a *direction* for its sense of smell**
   &mdash; not sharper sight. Going from one omnidirectional whiff to a five-cell smell field is
   worth **+45 survival steps**, the largest single-variable gain anywhere in the ladder.
   *(single contrast)*
2. **What matters about sight is *identity*, not sharpness.** Taking the agent's eight appearance
   channels down to one &mdash; so it sees *that* something is there but not *what* &mdash; costs
   **&minus;47 steps**, the largest single loss. Blurring the picture eightfold costs half as much.
   *(corroborated by finding 4)*
3. **Sharp sight is worse than slightly blurred sight (&minus;25 steps).** Not a paradox: with blur
   off, the agent sees an object only if it stands *exactly* on one of thirteen sampled cells.
   Section 3 gives the mechanism. *(single contrast)*
4. **An agent that can tell a rabbit from a predator stops hiding from rabbits.** All five arms
   whose sight cannot resolve identity hide *more* when a harmless rabbit is near; all nine that can
   resolve it hide *less*. Four independent measures separate the two groups with no overlap.
   *(pattern across all 14 arms)*
5. **The agent responds to what it FEELS, not to the wound it has.** It has no sensor for its own
   injury. It gets one delayed, smoothed trace of it from an interoceptive nociceptor, and that
   trace is **exactly zero for the first two steps** however badly hurt the agent woke up. The
   wound's own gap between the heaviest and lightest quarters is **largest at step 0** &mdash; where
   the behavioural response is nil, which the physical reading cannot explain and the perceptual one
   requires. The *perceived* gap peaks at **step 12** in all fourteen arms; extra hiding peaks at
   **step 14&ndash;16** in thirteen of them, two to four steps behind the feeling. The burst then
   reverses while the agent repays the 13&ndash;20 nutrition points its caution cost.
   *(all 14 arms agree on the timing; 13 of 14 on the response. An earlier version of this report
   told this story against the injury level, which the agent cannot sense &mdash; see Section 5.)*
6. **Hypervigilance shows up on the ambiguous channel &mdash; but weakly, and less uniformly than a
   smaller sample suggested.** A wound does *not* make the agent treat a nearby rabbit more like a
   nearby predator. It *does* amplify the response to a harmless animal's **smell** more than to a
   predator's smell, in **12 of the 14 arms**. The two exceptions are within noise of zero. But the
   clean group split that an earlier version of this report claimed for this measure **did not
   survive** tripling the evaluation sample &mdash; see the correction in Section 5.
   *(pattern across 12 of 14 arms; the effect is small, window-dependent, and its control channel is
   contaminated &mdash; the weakest finding here)*
7. **Hunger outweighs the wound &mdash; by 1.6&times; to 4.5&times; depending on the arm &mdash; and
   hiding is a cost, not a good.** Across the fourteen arms, more bush hiding goes with shorter life
   and less eating. *(pattern across arms; see the caveat in Section 8 about n = 14. Part of that
   ratio is sensor dynamics rather than drive strength: hunger reaches the agent instantly through
   satiation, while injury arrives through a twelve-step delay, so a 25-step window flatters hunger.)*

### The one caveat that colours everything below

**There is one training run per arm.** All fourteen were trained from the same random seed &mdash;
the number that fixes the network's initial weights and every random choice during training. Two
runs of the same configuration with different seeds land in different places, so a single run tells
you about *one* agent, not about the configuration. The 1,000,000 test worlds make each agent's
*behaviour* very precisely measured; they do nothing about the fact that each *agent* is one draw.

So a difference between two individual arms &mdash; "sharp sight costs 25 steps" &mdash; cannot be
separated from ordinary run-to-run variation, and is marked *single contrast* above.

A finding resting on a **pattern across many arms** carries more weight &mdash; but less than a
count of arms makes it look. The fourteen are emphatically **not** fourteen independent trials:

- They all replayed **bit-identical worlds**. Verified: the starting injuries, starting nutritions,
  animal counts and odour draws are exactly equal across all fourteen stores. So a quirk of the test
  worlds hits every arm the same way.
- They share **one training seed**, and the nine arms grouped together in finding 4 also share an
  identical observation vector width &mdash; which under one seed plausibly means identical initial
  weights and identical data ordering. The nine-against-five split coincides exactly with
  same-versus-different input width. Those nine are therefore closer to near-replicates of each
  other than to nine independent draws.

So "nine arms agree" must not be read as nine independent confirmations, and no claim in this
document multiplies arms together as if they were coin flips. What the patterns do establish is that
the effects are **systematic rather than idiosyncratic** &mdash; they are not one arm's quirk. What
they cannot establish is the effect *size* for a configuration, or that a differently-seeded ladder
would reproduce them. **Replicating at two or three seeds is the only thing that would move this
study from exploratory to confirmatory**, and until then every confirmatory-sounding sentence here
should be read as exploratory. No hypotheses were pre-registered.

An independent adversarial review of this analysis, including findings not yet acted on, is at
[`docs/reviews/plan_sensor_ladder.md`](../../../reviews/plan_sensor_ladder.md).

---

## Words this document uses

| term | what it means here |
|---|---|
| **arm** | one of the fourteen trained agents, together with the sensory configuration it was trained under |
| **ladder** | the set of fourteen, arranged so that most sit one single setting away from another |
| **bush hiding** | the share of an episode's steps the agent spent standing inside a bush. A bush hides it from predators and contains no food |
| **percentage point** (pp) | the arithmetic difference between two percentages. Going from 14% to 18% is +4 pp, not +29% |
| **a rate vs a difference** | a *rate* ("bush hiding, % of steps") is a share of something and can never be negative. A *difference* ("+4 pp") is one rate minus another and obviously can be. Several figures here plot differences, and each such axis says so; a negative value there means the agent hid **less** in one condition than the other, never that a percentage went below zero |
| **injury level** | the wound in the agent's body, 0-100. The agent has **no sensor for it** |
| **perceived nociception** | the one number the agent actually receives about its own injury: the last twelve injury levels convolved with an alpha kernel, the current step weighted zero, the buffer zeroed at reset. It lags the injury by several steps and is exactly zero for the first two steps of every episode |
| **appearance channel** | one of the numbers in the vector describing how a thing looks. With eight channels, food, rock, predator and rabbit each occupy a different one, so the agent can tell them apart. With one channel, everything it sees adds to the same number, so it registers *that* something is there and not *what* |
| **value mode** `sum` / `clamp` | how several objects in one cell combine. `sum` adds them, so two rocks read 2.0; `clamp` saturates at 1.0, so the agent sees presence but not count |
| **occlusion** | whether objects block the line of sight to things behind them. Which objects block is set per object type by a `blocks_sight` flag |
| **chebyshev distance** | the number of moves needed when diagonal moves are allowed &mdash; so a square two cells away diagonally is at distance 2. Used for everything the agent does, since it moves diagonally |
| **Manhattan distance** | the number of moves when diagonals are *not* allowed &mdash; the same diagonal square is at distance 4. Used only to describe the shape of the visual field, which is a Manhattan diamond |
| **neutral** / **rabbit** | the harmless animal class. `neutral` is its name in the config; this document calls it a rabbit |
| **hiding predator** / **ambush predator** | a stationary hazard that sits in the world and damages the agent if it steps on it. Distinct from the roaming predators that hunt |
| **active** animal | one that was placed in this particular episode. The world does not contain every animal every time: a third of episodes have no predator at all, and a third no rabbit |
| **nociceptor** | the agent's internal injury channel. It holds the last twelve injury *levels* (not damage events), zeroed at reset, and reports a weighted blend of them that emphasises the recent past and fades the older entries &mdash; a smoothed running trace of how hurt the body is |
| **quasi-binomial regression** | a model for a proportion &mdash; here, what fraction of an episode's steps were spent in a bush &mdash; that does not assume every step is an independent coin flip. Steps within one episode are obviously not independent, and this model allows for that |
| **overdispersion** | how much more variable the data are than the independent-coin-flip model would predict. It runs 13&ndash;27 here, meaning the data are over a dozen times more variable than that idealised model allows. Standard errors are multiplied by its square root; without that correction every p-value in this document would be far too small |
| **effect per standard deviation** | how much bush hiding moves when a feature moves by one standard deviation of its own spread across episodes &mdash; *not* by one unit. It lets features measured in different units be compared on one scale |
| **seed** | the number fixing the random initial weights and every random choice during training. All fourteen arms share one |

**Arm names.** The letter is the family and the suffix is the setting. `A` baseline, `B` olfaction,
`R` visual range, `V` visual blur, `P` blur anisotropy (shape), `Q` presence-only vision, `O`
occlusion. In the `V` family the digits are the blur scale with the decimal point removed:
`V4_blur05` is blur scale **0.5**, `V1_blur40` is **4.0**.

**Table 1.** the fourteen arms

| arm | what it can sense | smell grid | sight range | blur scale | blur anisotropy | appearance channels | value mode | sight blocked by |
|---|---|---|---|---|---|---|---|---|
| `A_baseline` | smell without direction | 0 | 0 | off | - | 8 | sum | nothing |
| `B_olf_only` | smell gains direction | 1 | 0 | off | - | 8 | sum | nothing |
| `R1_range1` | short-range sight | 1 | 1 | off | - | 8 | sum | nothing |
| `V1_blur40` | very blurred sight | 1 | 2 | 4.0 | 3.0 | 8 | sum | nothing |
| `V2_blur20` | blurred sight | 1 | 2 | 2.0 | 3.0 | 8 | sum | nothing |
| `V3_blur10` | mildly blurred sight | 1 | 2 | 1.0 | 3.0 | 8 | sum | nothing |
| `V4_blur05` | reference agent | 1 | 2 | 0.5 | 3.0 | 8 | sum | nothing |
| `V5_sharp` | sharp sight | 1 | 2 | off | - | 8 | sum | nothing |
| `P1_blur05_iso` | blur equal in all directions | 1 | 2 | 0.5 | 1.0 | 8 | sum | nothing |
| `Q1_presence_sum` | sight without identity | 1 | 2 | 0.5 | 3.0 | 1 | sum | nothing |
| `Q2_presence_binary` | sight without identity or count | 1 | 2 | 0.5 | 3.0 | 1 | clamp | nothing |
| `O1_occl_rock` | rocks block the view | 1 | 2 | 0.5 | 3.0 | 8 | sum | rock |
| `O2_occl_veg` | rocks and bushes block | 1 | 2 | 0.5 | 3.0 | 8 | sum | bush, rock |
| `O3_occl_all` | everything blocks | 1 | 2 | 0.5 | 3.0 | 8 | sum | bush, rock, predator, neutral, hiding_predator |

`P1_blur05_iso` changes only the blur's *anisotropy*. At the default of 3.0 the point-spread is
stretched along the line from the agent to the object and kept narrow across it, so distance becomes
vague while direction stays sharp. At 1.0 the spread is equal in every direction and direction
blurs too.

### What is causal here, and what is not

Three quantities are drawn **before the agent acts**, so splitting on them supports a causal claim:
the starting injury (uniform 0&ndash;100), the starting nutrition (uniform 0&ndash;100), and each
animal's odour, redrawn every episode.

Everything else is not. In particular, **distance to a predator is not randomised** &mdash; the
agent chose where to walk &mdash; so the distance curves in Figures 4, 5 and 10 are descriptive.
They describe a real regularity in behaviour; they do not by themselves establish that proximity
*caused* it.

### How many episodes each analysis actually uses

Not every analysis uses all 1,000,000. Stating this once, so no figure has to carry a footnote:

| analysis | episodes | why |
|---|---|---|
| survival, bush hiding, how it ends (Figures 1&ndash;3, 15) | 1,000,000 | everything |
| nearest-predator distance (Figures 4, 5, 10) | ~665,000 | a third of episodes contain no predator |
| nearest-rabbit distance (Figures 4, 5, 10) | ~665,000 | a third contain no rabbit |
| rabbit odour (Figures 11, 12) | ~665,000 | needs a rabbit to have an odour |
| odour inside a regression (Figure 7) | ~111,200 (11%) | needs *exactly* one predator and one rabbit, so each smell is one number rather than an average |
| the wound (Figures 8, 9, 13, 14) | 1,000,000 | every episode has a starting wound |

---

## 1. What each sense buys

![Survival and bush hiding for all fourteen arms](figures/lad01_ladder_overview.png)

**Figure 1.** Left: mean survival. Right: bush hiding. Both pooled over each arm's 1,000,000 episodes.

**Axes.** Both panels: y = the fourteen sensor-ladder arms, poorest senses at the bottom. x, left = mean survival in steps per episode; x, right = bush hiding as a percentage of an episode's steps. Both x-axes start at zero.

**Motivation.** The most basic question about a sense is whether having it helps. If it does, the
agents that have it should live longer.

**Method.** Survival is the mean episode length. Bush hiding is (bush steps) &divide; (steps), pooled
over every episode of the arm. The `t=0` row is the world as handed to the agent, not a step it
took, so it is excluded from both the numerator and the denominator.

**Reading.** Survival runs from 166 steps to 264 &mdash; the best arm lives about 1.6 times as long
as the worst, produced entirely by sensory settings. Bush hiding moves far less, 14.0% to 20.2%. The
two are *anti*-correlated: the agents that hide most are the ones that die soonest. That is the
first hint that hiding is not the thing the senses buy.

**Table 2.** outcome per arm

| arm | mean survival (steps) | bush hiding (%) | killed (%) | starved (%) | reached the limit (%) | food per 100 steps |
|---|---|---|---|---|---|---|
| `A_baseline` | 166.0 | 15.9 | 45.9 | 30.6 | 23.5 | 21.32 |
| `B_olf_only` | 211.3 | 20.2 | 47.3 | 24.4 | 28.3 | 20.84 |
| `R1_range1` | 230.9 | 16.4 | 39.8 | 27.8 | 32.4 | 25.81 |
| `V1_blur40` | 240.9 | 15.0 | 29.5 | 36.2 | 34.4 | 24.92 |
| `V2_blur20` | 250.3 | 14.3 | 29.8 | 34.2 | 36.1 | 27.34 |
| `V3_blur10` | 258.8 | 14.3 | 27.4 | 36.0 | 36.7 | 28.56 |
| `V4_blur05` | 264.2 | 14.0 | 25.6 | 36.9 | 37.6 | 29.55 |
| `V5_sharp` | 238.9 | 14.5 | 33.0 | 32.7 | 34.3 | 27.56 |
| `P1_blur05_iso` | 259.3 | 14.2 | 27.0 | 36.5 | 36.5 | 29.79 |
| `Q1_presence_sum` | 217.4 | 17.1 | 45.2 | 26.5 | 28.3 | 28.64 |
| `Q2_presence_binary` | 220.3 | 16.5 | 40.8 | 30.2 | 29.0 | 26.08 |
| `O1_occl_rock` | 253.3 | 14.3 | 27.3 | 36.9 | 35.8 | 28.39 |
| `O2_occl_veg` | 248.2 | 15.1 | 30.6 | 34.3 | 35.1 | 28.79 |
| `O3_occl_all` | 250.5 | 14.5 | 29.6 | 34.5 | 35.9 | 27.95 |

### Isolating one knob at a time

![Effect of each single-variable sensor change](figures/lad02_single_variable_steps.png)

**Figure 2.** Each row is one arm minus its reference arm &mdash; a change of exactly one setting.

**Axes.** Both panels: y = the twelve single-setting changes, each labelled with the arm it compares and the reference it is compared against. x, left = change in mean survival in steps; x, right = change in bush hiding in percentage points. Both x-axes are differences and are centred on zero, so a bar left of the line means the change made that quantity smaller.

**Motivation.** Figure 1's ranking confounds everything: the sharp-sighted agent differs from the
near-blind one in several settings at once. The ladder was built so that most arms sit one setting
away from a named reference, and those pairs are what Figure 2 shows.

**Method.** The difference of the two arms' pooled values from Figure 1. Because both members of a
pair were handed the same 1,000,000 worlds, they met identical predators, identical food and identical
starting wounds. The claim that each pair differs in exactly one setting is **asserted in code**
(`check_single_variable_pairs`), not merely intended &mdash; see the correction note below for why
that guard exists.

**Reading.** Three results stand out.

- **Direction beats acuity.** Giving smell a direction (`A_baseline` &rarr; `B_olf_only`) is worth
  **+45.3 steps** &mdash; the biggest single win in the table &mdash; even though the agent still
  has no useful sight at all.
- **Identity beats sharpness.** Collapsing the eight appearance channels to one (`V4_blur05` &rarr;
  `Q1_presence_sum`) costs **&minus;46.8 steps**. Blurring the picture eightfold (`V4_blur05` &rarr;
  `V1_blur40`) costs **&minus;23.3**. Seeing *that* something is there but not *what* is worse than
  seeing *what* very blurrily.
- **Sharp is worse than slightly blurred.** Disabling blur costs **&minus;25.3 steps**. Section 3
  explains why.

Halving the visual range, by contrast, costs only **&minus;8.0 steps** &mdash; much less than any of
the above.

> **Correction.** An earlier version of this document reported the visual-range change as
> &minus;33.5 steps, a figure four times too large. `R1_range1` was being compared against
> `V4_blur05`, which differs from it in **two** settings: the visual range *and* whether blur is on.
> The correct single-variable comparison is against `V5_sharp`, which also has blur off. The number
> above is the corrected one. The analysis code now asserts that every pair differs in exactly one
> setting and refuses to draw the figure otherwise; an audit under that rule found two further
> pairings that were also being over-counted.

**Table 3.** single-variable sensor changes

| change | arm | compared with | survival (steps) | bush hiding (pp) |
|---|---|---|---|---|
| olfactory grid range 1 - a 5-cell smell diamond | `B_olf_only` | `A_baseline` | +45.3 | +4.4 |
| visual range 1 (5 cells) instead of 2 (13 cells) | `R1_range1` | `V5_sharp` | -8.0 | +1.9 |
| visual blur radial scale 4.0 | `V1_blur40` | `V4_blur05` | -23.3 | +1.0 |
| visual blur radial scale 2.0 | `V2_blur20` | `V4_blur05` | -13.9 | +0.4 |
| visual blur radial scale 1.0 | `V3_blur10` | `V4_blur05` | -5.4 | +0.3 |
| visual blur disabled | `V5_sharp` | `V4_blur05` | -25.3 | +0.5 |
| visual blur anisotropy 1.0 instead of 3.0 | `P1_blur05_iso` | `V4_blur05` | -4.9 | +0.2 |
| visual vector size 1 - sees THAT, not WHAT | `Q1_presence_sum` | `V4_blur05` | -46.8 | +3.2 |
| visual value mode clamp instead of sum | `Q2_presence_binary` | `Q1_presence_sum` | +3.0 | -0.6 |
| visual occlusion on, rocks only | `O1_occl_rock` | `V4_blur05` | -10.9 | +0.4 |
| visual occlusion also by bushes | `O2_occl_veg` | `O1_occl_rock` | -5.1 | +0.7 |
| visual occlusion also by animals and ambush predators | `O3_occl_all` | `O2_occl_veg` | +2.3 | -0.6 |

---

## 2. What actually kills each agent

![How episodes end, per arm](figures/lad03_how_it_ends.png)

**Figure 3.** Every arm's 1,000,000 episodes split into the three ways an episode can end.

**Axes.** y = the fourteen arms, poorest senses at the bottom. x = share of that arm's episodes, 0 to 100 percent; the three coloured segments of each bar sum to 100 percent by construction.

**Motivation.** Survival alone hides the trade-off. An agent that hides constantly does not get
eaten &mdash; it starves. An agent that forages constantly does not starve &mdash; it gets eaten.
Which failure a sensory setting buys down, and which it buys up, is more informative than the net.

**Method.** The `termination_reason` column of the episode table, counted per arm. Three outcomes
occur here: reached the step limit alive, starved, killed by a predator. They sum to 100%.

**Reading.** Predation is where the senses pay. It runs from 46&ndash;47% in the two arms with no
useful sight down to 26% in the reference agent. Starvation moves the *other* way, from 24% up to
37% &mdash; the better-sighted agents trade some starvation risk for a much larger reduction in
predation, and come out ahead. `Q1_presence_sum` and `Q2_presence_binary` are the tell: they have a
full thirteen-cell visual field and are still eaten at 45% and 41%, essentially the rate of an agent
that cannot see at all. A visual field that cannot say *what* it contains does not protect against
being eaten.

---

## 3. Why sharp sight is worse than blurred sight

This is the one result in the report that looks like an error and is not, so the mechanism is worth
stating explicitly. In `sense_visual` (`src/environment/sensor.py`), the weight matrix that maps
objects onto the agent's sampled cells is built one of two ways:

- **Blur disabled** &mdash; an *exact cell match*. An object registers only if it stands exactly on
  one of the thirteen cells of the visual field. Those thirteen cells form a Manhattan diamond of
  radius 2, so a predator two cells away *diagonally* &mdash; Manhattan distance 4 &mdash; is
  completely invisible.
- **Blur enabled** &mdash; a *point-spread function*: instead of registering only on its own cell,
  each object contributes a Gaussian smear of signal to nearby cells, wider the further away it is,
  normalised so that only the fraction landing inside the diamond is reported. Distant objects fade
  rather than vanish.

So the blur setting is not an image-quality dial running from "good" to "degraded". It is a
**reach-versus-precision** dial. Turning it off buys perfect localisation of the few objects that
happen to sit on a sampled cell, at the cost of not seeing anything else at all.

The behaviour matches. **Figure 5** shows `V5_sharp` responding *less* strongly to a predator at one
or two cells (25.3 percentage points, against 34.1 for the reference agent), and **Figure 4** shows
it maintaining a *higher* level of hiding out at six and seven cells (12.2% against 8.0%). That is
the signature of an agent that frequently cannot see the predator right next to it and compensates
with a raised baseline everywhere. It also explains why `V5_sharp` is the arm that comes closest to
the false-alarm group on every odour measure in Table 5: with unreliable sight, it falls back on
smell.

And the ladder is non-monotonic in exactly the way a reach-versus-precision trade-off predicts:
survival runs 240.9 &rarr; 250.3 &rarr; 258.8 &rarr; **264.2** &rarr; 238.9 as the blur scale goes
4.0 &rarr; 2.0 &rarr; 1.0 &rarr; 0.5 &rarr; off. There is an optimum in the middle, at 0.5.

![Bush hiding against distance to the nearest animal](figures/lad04_threat_distance_curve.png)

**Figure 4.** Bush hiding against how far the nearest predator (left) or rabbit (right) was when the
agent chose its move. Colour is the grouping of Section 4; the four arms the text discusses are
drawn thick and named.

**Axes.** Both panels: x = distance from the agent to the nearest animal at the moment it chose its move, in chebyshev steps, from 1 (adjacent) to 8 or more. y = bush hiding as a percentage of those steps. Both panels share one y-scale. The y-axis does not start at zero; it is cropped to the range the curves occupy.

**Motivation.** Hiding is only defensive if the threat triggers it. A flat curve would mean the
agent hides on a schedule; a curve that rises as the animal approaches means it hides in response to
something it perceived.

**Method.** Distance to the nearest *active* predator, in chebyshev steps. The action that produced
step `t` was chosen while the agent was looking at step `t&minus;1`, so distance is read off the
previous row and bush occupancy off the current one. Distances of 8 or more are pooled into the
last bin. Episodes containing no predator are excluded from the predator panel &mdash; a third of
all episodes have none, and including them would silently make "there is no predator" the comparison
group.

**Reading.** Every arm hides far more with a predator close than with one far away. The rise is not
perfectly monotone at the very closest bin: in `B_olf_only`, `A_baseline` and four others, bush hiding
is *higher* at distance 2 than at distance 1. That dip is expected rather than puzzling &mdash; at
distance 1 the agent is often already caught in the open with a predator adjacent, which is a
different situation from spotting one approaching. The rabbit curves separate into two families,
which is the subject of the next section.

---

## 4. The rabbit false alarm, and what abolishes it

![Predator response against rabbit response, per arm](figures/lad05_discrimination.png)

**Figure 5.** Left: each arm's response to a nearby predator and to a nearby rabbit, joined by a line
whose length is that arm's discrimination. Right: the rabbit response alone, coloured by whether the
arm's sight can resolve identity.

**Axes.** Both panels: y = the fourteen arms, poorest senses at the bottom. x = a DIFFERENCE in percentage points — bush hiding when the animal is 1-2 cells away minus bush hiding when it is 6 or more cells away. Left panel shows that difference for a predator and for a rabbit; right panel shows the rabbit one alone. Below zero means the agent hides LESS when the animal is near.

**Motivation.** A rabbit cannot hurt the agent. Hiding when one comes near costs foraging time and
returns nothing. So the rabbit response is a clean measure of wasted defence &mdash; and asking which
sensory settings remove it tells us what discrimination actually requires.

**Method.** For each arm and each animal class, the *proximity effect* is

$$
P(\text{in bush} \mid \text{nearest animal 1-2 cells away}) \;-\; P(\text{in bush} \mid \text{nearest animal 6 or more cells away})
$$

in percentage points, over all steps. Counts are pooled before the ratio is taken, so a distance bin
holding more steps carries more weight; averaging the per-bin rates instead would let a sparse bin
dominate.

**Reading.** The fourteen arms split cleanly in two, and the split is **not** sight against no sight:

- **Nine arms whose sight can resolve identity** &mdash; visual range 2 with eight appearance
  channels &mdash; have a *negative* rabbit response, between &minus;1.0 and &minus;3.6 percentage
  points. They hide slightly *less* when a rabbit is near.
- **Five arms whose sight cannot** &mdash; the two with no useful visual range, the one with range 1,
  and the two with a single appearance channel &mdash; all have a *positive* rabbit response, between
  +2.7 and +7.6 percentage points.

`Q1_presence_sum` and `Q2_presence_binary` are the interesting cases. They have the full
thirteen-cell field at the reference blur; the only thing they lack is the ability to say what is in
it. They fall straight back into the false alarm.

**Table 4.** the rabbit false alarm, four independent measures

| arm | sight resolves identity | proximity to a rabbit (pp) | rabbit odour slope (pp) | rabbit odour, adjusted (pp) | one SD more rabbits (pp) |
|---|---|---|---|---|---|
| `A_baseline` | no | +3.2 | +8.22 | +3.81 | +3.62 |
| `B_olf_only` | no | +7.6 | +9.73 | +2.02 | +3.74 |
| `R1_range1` | no | +2.8 | +8.76 | +1.64 | +2.51 |
| `V1_blur40` | yes | -3.6 | +1.23 | +0.66 | +0.75 |
| `V2_blur20` | yes | -3.2 | +0.85 | +0.39 | +0.29 |
| `V3_blur10` | yes | -2.9 | +0.18 | -0.02 | -0.02 |
| `V4_blur05` | yes | -2.3 | +0.06 | -0.00 | +0.05 |
| `V5_sharp` | yes | -1.0 | +6.43 | +0.84 | +1.10 |
| `P1_blur05_iso` | yes | -2.6 | +0.20 | +0.09 | +0.01 |
| `Q1_presence_sum` | no | +3.3 | +9.04 | +1.83 | +3.07 |
| `Q2_presence_binary` | no | +6.0 | +9.93 | +1.88 | +3.13 |
| `O1_occl_rock` | yes | -2.8 | +3.21 | +0.69 | +0.65 |
| `O2_occl_veg` | yes | -3.1 | +3.35 | +0.70 | +0.47 |
| `O3_occl_all` | yes | -2.4 | +3.29 | +0.75 | +0.53 |

Four measures of the same false alarm, on two different cues, separate the two groups with **no
overlap** &mdash; the largest value among the nine never reaches the smallest among the five:

**Table 5.** does the split actually separate the two groups, or do they overlap?

| measure | largest value among the nine that resolve identity | smallest among the five that do not | separation | closest of the nine |
|---|---|---|---|---|
| how much it hides when a rabbit is near | -0.99 | +2.77 | +3.76 | `V5_sharp` |
| response to a strong rabbit smell | +6.43 | +8.22 | +1.79 | `V5_sharp` |
| rabbit smell, adjusted for the world | +0.84 | +1.64 | +0.80 | `V5_sharp` |
| one SD more rabbits in the world | +1.10 | +2.51 | +1.41 | `V5_sharp` |

**But read that with three qualifications, because it is easy to overstate.**

1. **The grouping rule is post-hoc.** "Sight resolves identity" is defined as visual range &ge; 2
   *and* more than one appearance channel. That rule was written after seeing which arms separated,
   not before. It is a description of the split, not a prediction of it, and nothing here was
   pre-registered.
2. **The four measures are not four independent experiments.** They are four views of the same
   behaviour stream from the same fourteen agents. Two are behavioural contrasts and two are
   regression coefficients, on two different cues (proximity and smell), which is real
   corroboration &mdash; but it is corroboration, not replication, and `V5_sharp` is the closest of
   the nine to the boundary on *all four*, with the smallest margin only 0.80 points.
3. **Identity is confounded with capacity.** Collapsing eight appearance channels to one does not
   only remove identity information; it shrinks the visual part of the observation from 104 numbers
   to 13. `Q1` and `Q2` therefore have a smaller network input as well as a less informative one.
   The ladder contains no arm that separates those two things, so "identity" is the most natural
   reading of this split, not the only possible one. `R1_range1` cuts the other way and is worth
   noting: it carries all eight identity channels and still shows the false alarm, which says range
   matters too.

![Which features of the world drive hiding, per arm](figures/lad06_world_factor_map.png)

**Figure 6.** Every feature of the world that is randomised before the agent acts, against every arm.
Red means the agent hides more; blue, less.

**Axes.** y = the nine features of the world that are randomised before the agent acts. x = the fourteen arms, poorest senses on the left. The colour of each cell, and the number printed in it, is the effect on bush hiding in percentage points of moving that feature by one standard deviation — red for more hiding, blue for less.

**Motivation.** Figures 1&ndash;5 each isolate one thing. This one steps back and asks which world
features move bush hiding at all, in every arm at once.

**Method.** A quasi-binomial regression per arm on the episode-level bush-hiding rate, with all nine
exogenous features entered together so each is adjusted for the others. Standard errors are scaled
by the Pearson overdispersion. The number plotted is the effect of moving the feature by one
standard deviation of its own spread, in percentage points of bush hiding &mdash; **not** the effect
of one more predator or one more rabbit. (One standard deviation of the rabbit count is 0.82
rabbits, so a per-rabbit effect would be about 22% larger than the number shown.) Only features
drawn at reset are included; consequences of the agent's own behaviour belong to a different
question and would swamp this one.

**Reading.** The `number of rabbits` row reproduces the Figure 5 split: +3.6, +3.7 and +2.5 for the
three arms without identity-resolving sight, +3.1 and +3.1 for the two presence-only arms, against
&minus;0.0 to +1.1 for the nine that can resolve identity. `number of predators` is the largest
driver in every single arm. It is *lower* in the identity-resolving arms (+9.3 to +12.4) than in
those without (+13.1 to +15.8), but not monotonically down the ladder &mdash; `B_olf_only` at +15.8
is the highest of all, above `A_baseline`. The row tracks identity-resolution, not "senses improving
in general": an agent that can tell what it is looking at needs less blanket caution because it can
afford to be selective.

> **Two rows in this figure look like they contradict the rest of the report, and here is why they
> do not.** `wound it woke up with` reads &minus;0.5 to +0.2 &mdash; essentially nothing &mdash;
> while Section 5 reports the wound moving bush hiding by up to +13 points. `how well fed it woke up`
> reads +0.6 to +1.7, against Section 6's much larger numbers. Both regressions here are fitted on
> the **whole-episode** bush-hiding rate, and both of those internal states are *transient*: the wound
> is 90% healed by step 28. Averaging over an episode of 250 steps dilutes a 30-step effect almost to
> nothing. Section 5 shows the full time course, and Section 7 shows exactly how much the choice of
> window changes the answer.

![The odour false alarm, adjusted](figures/lad07_odour_regression.png)

**Figure 7.** The effect of a predator's and a rabbit's odour strength on bush hiding, with the rest
of the world held fixed.

**Axes.** y = the fourteen arms, poorest senses at the bottom. x = the effect on bush hiding, in percentage points, of a one-standard-deviation stronger smell, with everything else in the regression held fixed. Centred on zero; right of the line means the agent hides more when that animal smells stronger.

**Motivation.** Figure 5 uses proximity. This uses smell, which is the genuinely ambiguous cue, and
puts it inside a regression so the result cannot be explained by strong-smelling worlds differing in
some other way.

**Method.** Quasi-binomial regression on the episode-level bush-hiding rate, restricted to the
&asymp;111,200 episodes per arm with exactly one predator and one rabbit, so that "the predator's
smell" and "the rabbit's smell" are each a single well-defined number rather than an average over
several animals. Adjusted for the number of bushes, rocks, food patches and ambush predators, the
distance the agent spawned from cover, the predator's detection range, attack delay, attack range
and stamina, the predator's own smell, and the agent's starting wound and hunger.

**Reading.** The same split, on a different cue and with the world held fixed. `A_baseline` hides
+3.97 points harder for a strong-smelling rabbit; the reference agent is at &minus;0.00 and not
distinguishable from zero.

---

## 5. What a wound does

### First: what the agent can actually feel

Everywhere else, a wounded agent is a suspicious comparison &mdash; it got hurt by doing something,
so its later behaviour is contaminated by whatever it was doing. This environment removes that
problem: the agent begins every episode with a wound drawn uniformly from 0 to 100 that it did
nothing to earn, so behaviour that tracks *that* number is caused by it.

But there is a second thing to get right, and an earlier version of this report got it wrong. **The
agent has no sensor for its injury level** &mdash; `injury_observable` is false in all fourteen arms.
What it receives is a single scalar from the interoceptive nociceptor: the last twelve injury levels
convolved with a normalised alpha kernel (tau = 3), with the current step's slot weighted **zero** so
nothing leaks in instantaneously, and with the whole buffer **zeroed at reset**.

That is not a technicality. An agent handed an injury of 100 feels:

| step | 0 | 1 | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|
| what it feels, as a share of the real wound | 0% | 0% | 9% | 36% | 61% | 79% | 100% |

It feels **nothing at all for two steps**, and does not feel the whole wound until step 12 &mdash; by
which time the wound itself has begun to heal. So "the wound" and "what the agent feels" are two
different signals with two different time courses, and an analysis that bins behaviour by the injury
level bins it by a quantity the agent cannot sense. That is what the earlier version did, and it is
why the timing claims in this section have been rewritten.

The perceived signal is reconstructed from each episode's recorded injury sequence exactly as the
environment builds it. The reconstruction was checked against the observation vector the environment
itself wrote &mdash; perceptual noise is disabled in this configuration, so the recorded channel *is*
the policy's input &mdash; and the worst disagreement over 1.36 million rows was **2.2 &times;
10⁻⁷**, which is float32 rounding.

![Bush hiding against the randomised starting wound](figures/lad08_injury_dose_response.png)

**Figure 8.** Bush hiding over each episode's first 25 steps, against the injury level the environment
handed the agent at `t=0`.

**Axes.** Left: x = the injury level the environment handed the agent at t=0, in four equal quarters of the 0-100 range; y = bush hiding over the episode's first 25 steps, as a percentage of those steps. Right: y = the fourteen arms; x = the difference between that arm's heaviest and lightest quarter, in percentage points. The left panel's y-axis does not start at zero; it is cropped to the range the curves occupy.

**Method.** Episodes are split into four equal quarters of the starting wound. Bush hiding is pooled
over the **first 25 steps only**. Figure 9 explains why 25, and Figure 14 shows what happens if you
choose otherwise.

**Reading.** Thirteen of the fourteen arms hide more when handed a bigger wound, by +2.6 to +6.7
percentage points across the full range. The exception is `A_baseline` at &minus;2.3, the one agent
with neither directional smell nor useful sight.

### The timing: behaviour follows the feeling, not the wound

![What the agent feels, and what it costs](figures/lad09_injury_time_course.png)

**Figure 9.** A: the injury level and the perceived signal, for the agents that woke in the lightest
and heaviest quarters. B: the three quantities as fractions of their own peak, so their timing can be
compared. C: bush hiding against how hurt the agent is feeling *right now*. D: the food the caution
cost. Faint lines are individual arms; bold lines are the two groups.

**Axes.** Panels A, B and D: x = step number within the episode, 0 to 100. y in A = injury level and perceived signal, both on the same 0-100 scale; y in B = each curve as a fraction of its own maximum, so only shape and timing are comparable, not size; y in D = the nutrition difference between the heaviest and lightest starting-wound quarters, on the 0-100 nutrition scale. Panel C: x = the perceived nociception the agent is receiving right now, in four equal quarters of the 0-100 range; y = bush hiding as a percentage of those steps. None of the four y-axes starts at zero; each is cropped to its own data.

**Motivation.** Figure 8 measures over 25 steps. Why 25? Widen the window and the effect shrinks: in
the reference agent it runs +4.0 percentage points at 25 steps, +0.7 at 50, and &minus;0.4 over the
whole episode. Taken at face value that looks like a result reported at a flattering window, which is
a fair thing to suspect. The answer is that the window is not tracking the wound &mdash; it is
tracking how long the agent can *feel* the wound.

**Method.** A step-by-step sweep records, for each arm and each quarter of the assigned starting
wound, the injury still carried, the reconstructed perceived signal, bush hiding, and nutrition at
every step to 120. Panels A, B and D are keyed to the **randomised** starting wound and are therefore
causal &mdash; the perceived signal is a deterministic function of that assigned wound, so keying to
one or the other does not change what is being manipulated, only what is being described. Panel C is
keyed to the **contemporaneous** perceived signal and is therefore associational: an agent feels hurt
because it got hurt, which depends on what it was doing. It is marked as such on the figure.

**Reading.** The timing settles it.

| quantity | peaks at | in how many arms |
|---|---|---|
| the injury gap &mdash; what the body is | **step 0** | 14 of 14 |
| the perceived gap &mdash; what the agent feels | **step 12** | 14 of 14 |
| extra hiding &mdash; what the agent does | **step 14&ndash;16** | 13 of 14 |

At step 0 the wound is at its most extreme &mdash; a 75-point gap between the quarters &mdash; and
the behavioural response is **nil**. Under the physical reading that is inexplicable. Under the
perceptual one it is required: the buffer is empty, so there is nothing to respond to. The response
then rises with the feeling and peaks two to four steps behind it. Across the arms, extra hiding
correlates with the perceived gap at r = +0.50 to +0.90 and with the injury gap at r = +0.72 to
+0.80; the correlations are close, but the *peak alignment* is not, and it is the peak alignment that
distinguishes the two accounts.

The exception is `A_baseline`, whose response is at noise level throughout (+1.05 pp peak, at step 4).
It is the one arm with neither directional smell nor useful sight.

**The size of the effect was also understated.** Binning by the assigned wound averages over the
whole ramp-up window in which the agent feels almost nothing, which is why Figure 8's numbers are
small. Binned by what the agent is actually feeling (panel C), bush hiding runs from **12.6&ndash;17.9%
at the lowest quarter of felt nociception to 31.7&ndash;49.5% at the highest** &mdash; a spread three
to five times larger than anything in Figure 8. That panel is associational, so it is not a
substitute for the causal contrast; but it is the right scale for the perceptual effect.

**And then the bill.** The hiding is paid for in food. An agent handed a heavy wound eats 4.4 food
items in its first 25 steps against 8.0 for one handed almost none, and runs 13&ndash;20 nutrition
points behind by around step 25. Once the feeling fades it hides **less** than the unhurt agent
&mdash; &minus;0.9 to &minus;4.7 points at step 60 &mdash; while it makes up the shortfall, and the
two converge by about step 100. Note that comparisons beyond roughly step 40 condition on survival,
and Table 8 shows the heavy-wound quarter dying sooner, so the late convergence is measured on
differently-selected survivors.

> **Correction.** An earlier version of this section said the extra hiding "falls away on roughly the
> wound's own schedule" and that 25 steps "is approximately the lifetime of the dose". Both were
> wrong, because both described the injury level rather than the perceived signal. The wound's gap is
> maximal at step 0 and half gone by step 17; the *felt* gap peaks at step 12 and is still around 30
> points at step 24. Twenty-five steps is roughly the lifetime of the **felt** dose, not the physical
> one. The causal claims are unaffected &mdash; they are keyed to the randomised starting wound
> either way &mdash; but the mechanism they were attached to was the wrong one.

**Table 10.** the wound's effect through time (Figure 9)

| arm | peak extra hiding (pp) | at step | extra hiding by step 60 (pp) | worst nutrition gap |
|---|---|---|---|---|
| `A_baseline` | +1.05 | 4 | -4.74 | -13.5 |
| `B_olf_only` | +10.50 | 14 | -2.42 | -15.4 |
| `R1_range1` | +11.35 | 16 | -1.59 | -16.2 |
| `V1_blur40` | +7.15 | 14 | -1.00 | -13.7 |
| `V2_blur20` | +7.15 | 14 | -1.13 | -17.1 |
| `V3_blur10` | +8.29 | 15 | -1.75 | -15.2 |
| `V4_blur05` | +9.29 | 15 | -1.39 | -18.1 |
| `V5_sharp` | +11.54 | 15 | -1.03 | -17.0 |
| `P1_blur05_iso` | +8.07 | 15 | -0.91 | -19.8 |
| `Q1_presence_sum` | +13.46 | 15 | -1.51 | -15.7 |
| `Q2_presence_binary` | +11.43 | 15 | -1.59 | -15.4 |
| `O1_occl_rock` | +12.85 | 15 | -1.59 | -14.5 |
| `O2_occl_veg` | +11.15 | 14 | -1.70 | -16.2 |
| `O3_occl_all` | +12.42 | 15 | -1.71 | -18.1 |

### But it is not hypervigilance &mdash; on this channel

![The wound's effect on the rabbit and predator responses](figures/lad10_hypervigilance_proximity.png)

**Figure 10.** Panels A and B: response to a nearby rabbit and to a nearby predator, at the lightest
and heaviest starting wound, **on one shared scale**. Panel C: the two shifts.

**Axes.** All three panels: y = the fourteen arms, poorest senses at the bottom. x in A and B = a DIFFERENCE in percentage points, bush hiding with the animal 1-2 cells away minus 6 or more cells away, drawn once for the lightest and once for the heaviest starting wound, and A and B share one scale. x in C = the difference between those two, in percentage points.

**Motivation.** Figure 8 shows a wounded agent hides more. That, on its own, is ordinary caution. The
claim that would earn the word *hypervigilance* is stronger and more specific: that being wounded
changes **what the agent treats as evidence of danger**, so an ambiguous, harmless cue starts driving
the same defence a real threat does. Testing that means comparing the shift in the harmless cue
against the shift in the real one. If a wound raised both by the same amount, the agent has simply
become more defensive across the board &mdash; a gain change, not a criterion change.

**Method.** Within each starting-wound quarter separately, the same proximity effect as Figure 5.
The shift is the heaviest quarter minus the lightest.

**Reading.** **No criterion shift on this channel.** The rabbit shift is between +0.0 and +0.8
percentage points in every arm. The predator shift is larger in most of them, up to +3.5. A wounded
agent becomes more responsive to the animal that can actually kill it &mdash; the opposite of what
hypervigilance predicts. Panels A and B share a scale so the rabbit response can be seen for what it
is: small.

### It *is* hypervigilance &mdash; on the ambiguous channel

![The wound's effect on the response to a harmless animal's smell](figures/lad11_hypervigilance_odour.png)

**Figure 11.** Left: how strongly each arm responds to a strong rabbit smell, having begun the
episode nearly unhurt or badly wounded. Right: the difference, with the predator smell as a control.

**Axes.** Both panels: y = the fourteen arms, poorest senses at the bottom. x, left = a DIFFERENCE in percentage points, bush hiding in the strongest-smelling quarter of episodes minus the weakest, drawn once for episodes begun unhurt and once for begun badly wounded. x, right = the difference between those two.

**Motivation.** Proximity is not the ambiguous cue. For an agent that can see, a nearby animal is a
*resolved* cue &mdash; it can look and tell what the animal is. Smell is the ambiguous one, and an
animal's odour is redrawn at random every episode. If a wound shifts the agent's criterion, this is
the channel where it should show.

**Method.** Episodes are split into quartiles of the odour intensity drawn for their rabbits. The
*slope* is bush hiding in the strongest-smelling quarter minus the weakest, over the first 25 steps,
computed once over episodes that began nearly unhurt and once over those that began badly wounded.
The bar is the difference. Episodes containing no rabbit are excluded.

**Reading.** In **12 of the 14 arms** the wound amplifies the rabbit-smell response more than the
predator-smell response; the gap runs &minus;0.34 to +1.67 points, and the two arms below zero
(`A_baseline` at &minus;0.02, `V1_blur40` at &minus;0.34) are small enough to be noise. The
predator-smell response *falls* with the wound in ten of the fourteen while the rabbit-smell
response rises in twelve, so the direction of the shift &mdash; toward the ambiguous channel &mdash;
is consistent even where the gap is small. The largest absolute amplification is in the three
occlusion arms (+1.42, +1.99, +1.71), where sight is present but scenery intermittently blocks it.

> **Correction, made when the evaluation sample was tripled.** At 300,000 episodes per arm this
> section reported the gap as positive in **all fourteen** arms, and claimed the rabbit-smell
> response grew *only* in the nine arms with identity-resolving sight (+0.18 to +2.00) while the
> other five were flat or falling (&minus;1.07 to +0.26). At 1,000,000 episodes neither holds. The
> gap is positive in twelve arms, not fourteen. And the absolute response now grows in **8 of the 9**
> with identity-resolving sight (&minus;0.37 to +1.99) **and in 4 of the 5 without**
> (&minus;0.73 to +0.67) &mdash; the two ranges overlap, so that split was an artefact of the smaller
> sample. What survives is the weaker, directional claim above. Nothing else in this report moved:
> every other headline number shifted by less than its rounding, and the four-measure split in
> Section 4 separates the groups as cleanly at 1M as it did at 300k.

**Two caveats that keep this from being as clean as it looks.**

- **The control channel is not neutral.** Figure 12 shows that the loudest-smelling predators are
  also the *least* distinguishable ones, because odour intensity is the sum of two channels while
  predator-ness is their difference. That mechanically flattens the predator slope &mdash; which is
  the quantity the rabbit slope is being compared against. Some of the rabbit-minus-predator gap is
  therefore an artefact of how intensity is defined, not a fact about the agent.
- **It is window-dependent, like everything else about the wound.** Measured over 50 steps instead of
  25, the gap goes negative in several arms. Given Figure 9, that is expected &mdash; the wound is
  gone by then &mdash; but it means the finding is a statement about the wounded phase, not about the
  episode.

Taken with those caveats, the reading this supports is a modest one: a wound does not make the agent
generically more afraid, and what shift there is goes toward the **ambiguous** channel rather than
the resolved one. The mechanism suggested by the occlusion arms &mdash; that an agent leans hardest
on smell when its sight is unreliable &mdash; is consistent with their being the largest in absolute
terms, but the ladder was not built to test that and the effect is small. On the
rabbit-minus-predator gap the leaders are `Q2_presence_binary` (+1.67) and `V5_sharp` (+1.62), which
are *not* occlusion arms, so "largest in the occlusion arms" holds for the absolute measure and not
for the gap. This is the least secure finding in the report and the one most in need of the
seed replication.

![Bush hiding against the randomised odour draw](figures/lad12_odour_false_alarm.png)

**Figure 12.** The underlying curves: bush hiding against how strongly this episode's rabbits (left)
and predators (right) happened to smell. Bold lines pool the counts within each group; faint lines
are the fourteen individual arms.

**Axes.** Both panels: x = how strongly that episode's animals happened to smell, in four quartiles of the randomised odour draw, weakest on the left. y = bush hiding over the episode's first 25 steps, as a percentage of those steps. Both panels share one y-scale. The y-axis does not start at zero; it is cropped to the range the curves occupy.

**Reading.** The rabbit panel rises for both groups and much more steeply for the five that cannot
resolve identity. The predator panel dips at the loudest quartile, and that dip is real rather than
an artefact: intensity is the **sum** of the two odour channels while what marks an animal out as a
predator is their **difference**, so an episode whose predators smell very loudly has both channels
near their ceiling and the difference squeezed toward zero. Measured on the reference run, mean
predator-ness is 0.12 in the loudest quartile against 0.18&ndash;0.22 in the other three. The loudest
predators are the least distinguishable ones, and the agent responds to them less &mdash; which also
means a loud rabbit is genuinely harder to rule out, so part of the "false" alarm is rational.

**Table 6.** what a randomised starting wound does

| arm | bush hiding, first 25 steps (pp per full wound range) | shift in rabbit proximity (pp) | shift in predator proximity (pp) | wound amplifies rabbit odour (pp) | wound amplifies predator odour (pp) |
|---|---|---|---|---|---|
| `A_baseline` | -2.34 | +0.01 | +3.52 | -0.73 | -0.70 |
| `B_olf_only` | +3.38 | +0.06 | -0.33 | +0.67 | -0.32 |
| `R1_range1` | +4.15 | +0.14 | -0.61 | +0.07 | -1.33 |
| `V1_blur40` | +2.63 | +0.55 | +1.32 | -0.37 | -0.04 |
| `V2_blur20` | +2.57 | +0.47 | +0.97 | +0.06 | -0.23 |
| `V3_blur10` | +3.46 | +0.47 | +1.19 | +0.70 | -0.59 |
| `V4_blur05` | +4.00 | +0.30 | +0.84 | +0.44 | +0.07 |
| `V5_sharp` | +5.38 | +0.73 | +0.93 | +1.06 | -0.56 |
| `P1_blur05_iso` | +3.59 | +0.47 | +1.59 | +0.66 | -0.02 |
| `Q1_presence_sum` | +5.22 | +0.41 | -0.22 | +0.31 | -0.84 |
| `Q2_presence_binary` | +4.84 | +0.64 | +0.25 | +0.36 | -1.31 |
| `O1_occl_rock` | +6.69 | +0.46 | +1.54 | +1.42 | +0.83 |
| `O2_occl_veg` | +5.12 | +0.76 | +1.14 | +1.99 | +1.07 |
| `O3_occl_all` | +5.41 | +0.70 | +0.81 | +1.71 | +0.90 |

---

## 6. The wound competes with hunger, and loses

![Bush hiding against starting hunger and starting wound](figures/lad13_two_internal_drives.png)

**Figure 13.** Bush hiding over the first 25 steps against the two internal states the environment
assigns at random, on a shared y-scale.

**Axes.** Both panels: x = the internal state the environment assigned at t=0, in four equal quarters of the 0-100 range — nutrition on the left (so the left end is an agent that woke starving) and injury on the right. y = bush hiding over the episode's first 25 steps, as a percentage of those steps. Both panels share one y-scale so their slopes can be compared directly. The shared y-axis does not start at zero; it is cropped to the range the curves occupy.

**Motivation.** The agent carries two internal states that pull in opposite directions. A wound
argues for staying in cover; an empty stomach argues for leaving it, because a bush contains no food.
Both are randomised at reset, so both can be tested causally and on the same footing.

**Method.** Episodes are split into four equal quarters of the assigned value. Both panels use the
first 25 steps, for the reason Section 5 gives. **Note the direction of the hunger axis**: it plots
*nutrition*, so the left end is an agent that woke up starving and the right end one that woke up
well fed. The slope is positive, meaning a **well-fed** agent hides more &mdash; equivalently, a
hungry one hides less, because it has to go and eat.

**Reading.** Nutrition moves bush hiding by +6.5 to +19.2 percentage points across its range; the
wound moves it by &minus;2.3 to +6.7. The ratio runs from 1.6&times; to 4.5&times; depending on the
arm. The metabolic drive is the larger of the two in every arm, and any account of this agent's
hiding that leaves it out is describing a small part of the behaviour.

**Table 9.** the two internal drives, first 25 steps

| arm | hunger: change in bush hiding (pp, signed) | wound: change in bush hiding (pp, signed) | ratio |
|---|---|---|---|
| `A_baseline` | +9.16 | -2.34 | wound effect is negative |
| `B_olf_only` | +15.12 | +3.38 | 4.5x |
| `R1_range1` | +18.34 | +4.15 | 4.4x |
| `V1_blur40` | +7.98 | +2.63 | 3.0x |
| `V2_blur20` | +9.66 | +2.57 | 3.8x |
| `V3_blur10` | +9.39 | +3.46 | 2.7x |
| `V4_blur05` | +6.48 | +4.00 | 1.6x |
| `V5_sharp` | +11.77 | +5.38 | 2.2x |
| `P1_blur05_iso` | +8.25 | +3.59 | 2.3x |
| `Q1_presence_sum` | +19.15 | +5.22 | 3.7x |
| `Q2_presence_binary` | +14.64 | +4.84 | 3.0x |
| `O1_occl_rock` | +12.06 | +6.69 | 1.8x |
| `O2_occl_veg` | +9.89 | +5.12 | 1.9x |
| `O3_occl_all` | +9.97 | +5.41 | 1.8x |

---

## 7. Two ways to get the injury result wrong

![Three readings of the same injury question](figures/lad14_window_and_variable.png)

**Figure 14.** The same question answered three ways, on one shared y-scale. A: the assigned wound
over the first 25 steps. B: the assigned wound over the whole episode. C: the carried wound over the
whole episode.

**Axes.** All three panels: x = an injury level in four equal quarters of the 0-100 range — the wound assigned at t=0 in panels A and B, the wound the agent was carrying when it decided in panel C. y = bush hiding as a percentage of steps. All three share one y-scale, which is the point of the figure. The shared y-axis does not start at zero; it is cropped to the range the curves occupy.

**Motivation.** "Does injury make the agent hide?" has three plausible-looking answers in this data
and two of them are wrong. They are shown together because the two mistakes have *different* causes
that are easy to conflate, and because a reader who reproduced either one deserves to know why it
differs.

**Method.** All three use identical bins and the identical outcome, bush hiding with the `t=0` row
excluded. They differ only in which injury number a step is filed under, and over which steps the
average is taken. All three panels share one y-axis, so the sizes are directly comparable.

**Reading.**

- **Panel A** is the honest measurement: the randomised wound, measured while the assigned dose is
  still largely present. Positive in thirteen of fourteen arms.
- **Panel B** changes *one* thing &mdash; the same randomised wound, averaged over the whole episode
  &mdash; and the answer collapses to between &minus;2.0 and +0.0. Section 5 explains why: the
  wound heals by step 28, so an average over 250 steps is mostly measuring an agent with no wound,
  plus the compensatory foraging that follows. The panel looks nearly flat on the shared scale, which
  is the correct impression.
- **Panel C** changes the *variable* instead &mdash; the wound the agent was carrying mid-episode,
  which is what falls out of any trajectory log for free &mdash; and reports **+9 to +28 points**.
  This is the most misleading of the three, because a mid-episode wound is a *consequence* of
  behaviour: the agent is carrying a big wound precisely because it was out in the open near a
  predator, which is also where the bushes are not. Panel C measures where the agent *was* and
  reports it as what the agent *decided*.

The spread between panel A and panel C &mdash; +4.0 against +18.7 in the reference arm &mdash; is the
size of the mistake available to anyone who takes the convenient measurement.

**Table 7.** the three readings of the injury question (Figure 14)

| arm | A: assigned wound, first 25 steps (pp) | B: assigned wound, whole episode (pp) | C: carried wound, whole episode (pp) |
|---|---|---|---|
| `A_baseline` | -2.34 | -1.98 | +9.03 |
| `B_olf_only` | +3.38 | -0.76 | +27.32 |
| `R1_range1` | +4.15 | -0.49 | +28.00 |
| `V1_blur40` | +2.63 | -0.49 | +19.08 |
| `V2_blur20` | +2.57 | -0.55 | +21.25 |
| `V3_blur10` | +3.46 | -0.51 | +16.67 |
| `V4_blur05` | +4.00 | -0.41 | +18.65 |
| `V5_sharp` | +5.38 | -0.17 | +23.78 |
| `P1_blur05_iso` | +3.59 | -0.42 | +17.27 |
| `Q1_presence_sum` | +5.22 | -0.49 | +25.56 |
| `Q2_presence_binary` | +4.84 | -0.32 | +24.41 |
| `O1_occl_rock` | +6.69 | +0.00 | +19.64 |
| `O2_occl_veg` | +5.12 | -0.36 | +21.99 |
| `O3_occl_all` | +5.41 | -0.37 | +21.88 |
**Table 8.** how long an episode lasts, by the wound the agent woke up with

| arm | started 0-25 | started 25-50 | started 50-75 | started 75-100 | difference |
|---|---|---|---|---|---|
| `A_baseline` | 176.6 | 173.1 | 163.8 | 150.4 | -26.2 |
| `B_olf_only` | 222.9 | 219.1 | 209.2 | 194.1 | -28.8 |
| `R1_range1` | 235.9 | 234.4 | 230.4 | 222.6 | -13.3 |
| `V1_blur40` | 247.2 | 244.8 | 240.3 | 231.1 | -16.0 |
| `V2_blur20` | 255.5 | 253.3 | 249.5 | 242.8 | -12.7 |
| `V3_blur10` | 264.1 | 262.2 | 258.1 | 250.8 | -13.4 |
| `V4_blur05` | 269.9 | 267.9 | 263.3 | 255.7 | -14.2 |
| `V5_sharp` | 243.6 | 241.7 | 238.1 | 232.0 | -11.7 |
| `P1_blur05_iso` | 265.3 | 262.7 | 258.6 | 250.6 | -14.6 |
| `Q1_presence_sum` | 223.0 | 220.6 | 216.1 | 209.8 | -13.2 |
| `Q2_presence_binary` | 226.5 | 223.9 | 218.8 | 212.0 | -14.6 |
| `O1_occl_rock` | 258.0 | 256.6 | 252.6 | 246.0 | -12.0 |
| `O2_occl_veg` | 253.1 | 251.2 | 247.6 | 240.9 | -12.2 |
| `O3_occl_all` | 255.4 | 253.4 | 249.7 | 243.2 | -12.2 |

---

## 8. What hiding costs

![Bush hiding against survival and against eating](figures/lad15_price_of_hiding.png)

**Figure 15.** One point per arm, over that arm's 1,000,000 episodes.

**Axes.** Both panels: x = bush hiding as a percentage of an episode's steps, one point per arm. y, left = mean survival in steps per episode; y, right = eating rate in food items per 100 steps. Neither axis starts at zero; both are cropped to the range the fourteen points occupy.

**Motivation.** If hiding were simply good, the arms that hide most would be the ones that survive
longest. Testing that directly is the cleanest way to say what the senses are actually for.

**Method.** Bush hiding is bush steps over steps; eating rate is `ate_food` events per 100 steps;
survival is mean episode length. The dashed line is a least-squares fit across the fourteen
**arm-level** points, and `r` is the correlation across those fourteen points &mdash; not across
1,000,000 episodes.

**Reading.** More hiding goes with **shorter** life (r = &minus;0.62) and **less** eating
(r = &minus;0.70).

**How much weight that correlation can bear: not much on its own.** It is computed on n = 14 points
that are not independent (see the caveat in the Question section), no confidence interval is given,
and it is visibly driven by `A_baseline` and `B_olf_only`, which sit detached in the lower right of
both panels. Cover those two and the remaining twelve show a much weaker relationship. What the
figure establishes is the *direction* &mdash; and the direction is corroborated independently by
Figure 3, where predation and starvation move in opposite directions across the ladder, and by
Figure 9, where extra hiding is followed by a measured food debt in every arm. The claim rests on
those, with this figure as the summary picture rather than the evidence.

The through-line of the whole report is there: the senses do not make the agent hide more.
`A_baseline` and `B_olf_only` hide the most and die soonest. Better senses let the agent hide at the
*right* moments and forage the rest of the time. What improves is not the amount of defence but its
**selectivity**.

---

## Method summary

**Data.** Fourteen `lad_*` training runs, all finished at 10,000,005&ndash;10,000,081 environment
steps (a spread of 76, so no arm had a meaningfully longer training budget). All share seed 42 and
one environment configuration; only the `sensory` block and the per-object `blocks_sight` flags
differ. Each arm's final checkpoint was run on 1,000,000 evaluation worlds seeded from a common base,
producing 166&ndash;264 million step rows per arm and about 3.32 billion in total.

**Pairing.** All fourteen arms were handed the *same* seed range, and this was verified rather than
assumed: the starting injuries, starting nutritions, animal counts and odour draws are exactly equal
across all fourteen stores. Between-arm differences are therefore free of world variance. They are
**not** free of training variance &mdash; see the limitations.

**Two conventions every script obeys.**

1. **The `t=0` row is not a step.** It is the world as handed to the agent. It appears in no numerator
   and no denominator. (Getting this wrong in an earlier analysis produced episodes in which the
   number of bush steps exceeded the number of steps &mdash; the count that was supposed to be a
   subset was larger than the set.)
2. **Predictors come from the previous row.** The action that produced row `t` was chosen while the
   agent was looking at row `t&minus;1`, so anything the agent conditioned on is read off `t&minus;1`.

**Regressions.** Quasi-binomial on the episode-level bush-hiding rate, with standard errors scaled by
the Pearson overdispersion. Effects are reported per standard deviation of the regressor.

**Three corrections made during this analysis**, recorded because each would have put a wrong number
into circulation:

1. **A third of episodes contain no predator, and a third no rabbit.** In an early version those were
   not excluded from the distance and odour analyses. Because `numpy.digitize` files every missing
   value into the *top* bin and `numpy.clip` files every infinite distance into the *farthest* bin,
   the no-predator episodes &mdash; which hide far less, since nothing is hunting them &mdash;
   silently became the comparison group. That manufactured a large spurious drop in the top odour
   quartile and inflated every proximity effect.
2. **One reference pairing was not single-variable.** `R1_range1` was compared against `V4_blur05`,
   which differs from it in two settings, inflating the reported cost of a shorter visual range from
   &minus;8.0 steps to &minus;33.5. The code now asserts single-variable pairing and refuses to draw
   the figure otherwise.
3. **The whole-episode injury result was explained by the wrong mechanism.** An earlier draft
   attributed it to episode-length weighting. The decomposition in Figure 9 shows the real cause: the
   wound heals, and the agent then repays a food debt. The earlier explanation predicted the wrong
   sign.

## Limitations

1. **One training run per arm, and the arms are not independent.** All fourteen share seed 42 and
   replayed bit-identical worlds; the nine arms grouped together in Section 4 additionally share an
   observation vector width, so under one seed they plausibly share initial weights and data
   ordering. Counts of agreeing arms must not be read as independent confirmations. **Replicating
   the ladder at two or three seeds is the single change that would move this study from exploratory
   to confirmatory.** Nothing here was pre-registered.

   **Decided 2026-09-01 — deferred, not abandoned.** This study's role is a **design-rationale
   appendix**: it justifies the sensor configuration the project trains on, and carries no headline
   claim of its own. The seed replication is therefore not blocking, and will be run once, later, on
   fixed trainer code — not now. The reason for waiting is concrete rather than budgetary: a
   Monte-Carlo return-units bug, already measured and with a fix drafted, under-credits roughly 35%
   of training targets in every run of this trainer, these fourteen arms included. Replicating today
   would produce twenty-eight more agents trained on code the project has already decided is wrong.
   The trigger for the deferred replication is that fix landing.

   Recorded because it goes against the project's own internal advice: the Principal Investigator's
   read was that the ladder is *already* the control arm for the main paper — the pain-like
   signatures are defined as exceeding what a pure-nociception agent produces, and these
   modulator-off agents are the only measurement of that agent we have — which would have made
   replication mandatory rather than optional. The decision went the other way. If the ladder is
   later cited as a control rather than as design rationale, this limitation becomes blocking again
   and the replication has to happen first. See `docs/pi/calls/2026-09-01_sensor_ladder_replication.md`.
2. **Everything about the wound is window-dependent, because the wound is transient.** Figure 9 makes
   the time course explicit and shows that the 25-step window matches the dose's lifetime, but any
   single number quoted for the wound is a statement about a window, and the hypervigilance gap in
   Section 5 changes sign in several arms at 50 steps.

   **This applies to the dose-response itself, not only to the hypervigilance measure**
   (added 2026-09-03). Figure 14 panel B already shows it; stating it here because the limitations
   list is what a hurried reader reads. Recomputed at three windows: the wound's effect on bush hiding
   is positive in 13 of 14 arms over 25 steps, in 2 of 14 over 100 steps, and no larger than
   +0.003 percentage points in any arm over the whole episode. So "this agent hides more when
   injured" is a claim about a 25-step window and must always be quoted with it.
3. **The 25-step "extra hiding" is extra RESTING that lands in cover, not travel to cover**
   (added 2026-09-03; this limitation qualifies finding 5). A wound raises the Rest-action rate by
   +17.7 to +35.2 percentage points in 14 of 14 arms — roughly five times the size of the bush-hiding
   effect — and that extra resting happens overwhelmingly *in the open*. Decomposed, bush occupancy
   **while the agent is acting** falls with the wound in 14 of 14 arms, in both the predator-present
   and predator-absent conditions. Section 5's reading as "ordinary caution" is therefore too strong:
   the wounded agent freezes to heal, and with four to ten bushes scattered over a hundred squares a
   slice of that freezing lands on one. Details, and the reconciliation with the project's separate
   bush-refuge finding that injury *suppresses* cover use, are in
   [`INJURY_HIDING_SIGN_RECONCILIATION`](../diagnosis/INJURY_HIDING_SIGN_RECONCILIATION.md).
4. **Proximity is not randomised.** The distance curves in Figures 4, 5 and 10 describe a real
   regularity but do not on their own establish causation.
5. **"Identity-resolving sight" is a post-hoc, two-condition proxy** that also confounds identity
   with input capacity. See the three qualifications in Section 4.
6. **The predator-odour control is contaminated.** Intensity and discriminability are anticorrelated
   by construction, which flattens the control slope and inflates the rabbit-minus-predator gap in
   Section 5.
7. **Hypervigilance was tested on two channels, not all of them.** The agent also has collision,
   proprioceptive and visual channels that were not tested for a criterion shift.
8. **The hunger-versus-wound comparison is partly a comparison of sensors.** Nutrition is not
   directly observable either, but the channel that carries it &mdash; satiation &mdash; is an
   instantaneous monotone transform of it, with no delay. Injury reaches the agent through a
   twelve-step convolution. Over a 25-step window that difference alone favours hunger, so the
   1.6&ndash;4.5&times; ratio in Section 6 mixes drive strength with sensor dynamics and should not
   be read as the former alone.
9. **The agent had an incentive to infer its wound faster than it can feel it.** The training reward
   uses the *true* injury level, so a policy that inferred its condition from context could in
   principle have beaten the nociceptor's delay. Empirically it did not &mdash; the response tracks
   the delayed percept, not the wound &mdash; but that is an observation, not a constraint.
10. **Late-time comparisons condition on survival.** Beyond roughly step 40, Figure 9 compares
   whichever episodes are still alive, and Table 8 shows the heavy-wound quarter dying sooner. The
   convergence after step 60 is therefore measured on differently-selected survivors.

## Reproducing this

Three stages. The analysis stage was always scripted; the two before it were not, and were
reconstructed on 2026-09-01 &mdash; see the note at the end of this section.

**1. Train the fourteen agents** (~11 h each, run in parallel across seven nodes):

```bash
bash scripts/lab/launch_ladder_arm.sh <arm> <cuda_index> <node_label>
```

One call per arm, with `<arm>` one of the fourteen config stems in
`configs/environment/experiment/sensory_ladder/`. Seed is **not** a parameter: the sweep is
single-seed by design (seed 42, from `configs/train/default.yaml`) and compares across arms, not
across seeds. Limitation 1 above is about exactly this.

**2. Collect the evaluation population** (1,000,000 episodes per arm, seeds 1,000,000&ndash;1,999,999,
~452 GB in total), in two contiguous passes:

```bash
P=/home/vncuser/miniconda3/envs/grid_world_pain/bin/python
$P scripts/eval/traj_collect/run_collection.py configs/trajectory_collection/sensor_ladder_pass1.yaml
$P scripts/eval/traj_collect/run_collection.py configs/trajectory_collection/sensor_ladder_pass2.yaml
```

Two passes rather than one because `n_episodes` is a guarded field of a store's manifest: the
300,000-episode store of pass 1 cannot be reopened and extended to a million, so pass 2 is a second
store covering the remaining 700,000 seeds. All fourteen arms share a seed base, so they are
**paired** &mdash; episode *i* faces the same environment draw in every arm.

**3. Run the analysis:**

```bash
P=/home/vncuser/miniconda3/envs/grid_world_pain/bin/python
$P scripts/analysis/studies/sensor_ladder/collect_arm_data.py     # one sweep per arm, ~60-95 s each
$P scripts/analysis/studies/sensor_ladder/collect_time_course.py  # the step-by-step sweep for Figure 9
$P scripts/analysis/ladder/run_all.py              # all fifteen figures, seconds
$P scripts/analysis/ladder/make_report_tables.py   # every table in this document
$P scripts/analysis/ladder/build_artifact.py       # the shareable HTML page
```

Every figure has exactly one script, named for its figure number, and each script states its own
question, method and known limitations in its docstring.

**A reproducibility gap, closed late and honestly.** Until 2026-09-01 this section covered only
stage 3. The fourteen arm configs and the launch script existed **only as untracked files** on the
NAS &mdash; not ignored, simply never added &mdash; so a published study could not be rebuilt from a
fresh clone. They are now committed, and were verified before committing against the config the
trainer itself saved beside each arm's checkpoints: every leaf value agrees, 14 of 14. The two
collection specs in stage 2 were never saved at all; they are **regenerated from the `_manifest.json`
each store carries**, which records every resolved parameter. Re-running them reproduces the
population, not the original invocation, which is unrecoverable. Both were checked with the
driver's `--dry-run`. Figures 6 and 7 additionally require
`scripts/analysis/hiding_drivers.py` to have been run per arm. See
[`scripts/analysis/ladder/README.md`](../../../../scripts/analysis/ladder/README.md).

## Related

- [`INJURY_HIDING_SIGN_RECONCILIATION`](../diagnosis/INJURY_HIDING_SIGN_RECONCILIATION.md) &mdash;
  reconciles this study's "hides more when injured" (Figure 8) with the project's separate
  bush-refuge finding that injury *suppresses* cover use. Verdict: not a contradiction &mdash;
  different agents, a different world, a different measurement scene, and a different window. It
  also identifies the behaviour underneath both, a wound-driven **freeze-and-rest** response, and
  is the source of Limitations 2 and 3 above.
- [`plan_sensor_ladder`](../../../reviews/plan_sensor_ladder.md) &mdash; the adversarial review of
  this analysis, including findings not yet acted on.
- [`a01_hiding_drivers`](../trajectory_factors/a01_hiding_drivers.md) &mdash; the same factor
  analysis on the ten `rppo_restprem` arms, which share one sensory setting and vary the agent
  instead.
- [`artifact_generation_guide`](../../../develop/active/meta/artifact_generation_guide.md) &mdash;
  the checklist these figures were built against.
