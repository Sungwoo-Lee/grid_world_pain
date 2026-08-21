# Which parts of the environment make the agent hide?

**Shareable page:** https://claude.ai/code/artifact/40bd0021-9f74-4df6-84cf-7e523c9d31a4
**Figures + per-point data:** `results/analysis/trajectory_glm/` (gitignored)
**Reproduce:** `scripts/analysis/trajectory_glm.py`

## Question

We put one trained agent back into the exact world it learned in, ran it for a million
episodes, and asked which features of that world change how much it hides in a bush. This is
the first use of the trajectory-collection pipeline
([[TRAJECTORY_COLLECTION_PIPELINE]]) and doubles as its end-to-end check.

The agent is **a01** — the arm where resting gives no healing bonus for continuing to rest
(the zero-premium anchor of the rest-premium sweep). One agent only; this is a pipeline
check, not the full comparison across arms.

The environment rerolls its own contents at the start of every episode: how many predators
appear (0, 1 or 2), how far each can see, how long it pauses before striking, how far it can
pounce, how much stamina it has, how many bushes exist, and how many hidden ambush predators
are buried in the map. **Because those rerolls are random and independent of the agent, their
effects are causally identified** — this is a randomised experiment the environment runs on
itself, a million times.

## Headline finding

**The obvious outcome variable gives the answer backwards.**

Counting *total steps spent in a bush* reverses the sign on the strongest factor in the
environment. As a predator's detection range rises from 1 to 7:

| | range 1 | range 7 |
|---|---|---|
| share of episode hiding | 20.0% | **56.7%** |
| total bush-dwell steps | 48.6 | **25.6** |
| episode length (survival) | 242.5 | **45.1** |

Both are true. The agent hides far more *intensely* and far less *in total*, because a
predator that sees further kills it five times sooner. Total steps is mostly a measurement of
how long the agent stayed alive, not of what it did.

So the analysis uses a **rate** — bush steps out of episode length, fitted as a binomial GLM
— and reports **survival separately as an outcome in its own right**. Survival is not a
nuisance to divide out; it is the quantity that makes the other two measures disagree.

This is the second time in two days that a raw-versus-normalised confusion has pointed the
wrong way here; the first was the ambush-risk analysis, recorded in the LLM wiki.

## Results

Change from the lowest to the highest level of each factor.

| factor | hiding % | total steps | survival |
|---|---|---|---|
| **number of predators** (0→2) | 7.9 → **56.7** | 33 → 29 | 416 → **51** |
| **detection range** (1→7) | 20.0 → **56.7** | 49 → 26 | 243 → **45** |
| attack delay (1→3) | 30.7 → 33.6 | 27 → 38 | 89 → 114 |
| attack range (2→3) | 32.9 → 31.5 | 36 → 30 | 109 → 96 |
| **max stamina** (30→150) | 32.3 → 32.3 | 33 → 33 | 102 → 102 |
| bushes available (4→10) | 11.9 → 20.4 | 22 → 40 | 184 → 195 |
| ambush predators (2→12) | 16.3 → 17.2 | 33 → 30 | 204 → 175 |

Notes on individual factors:

- **Predator count** is the largest effect in the environment, and **detection range** is
  nearly as large on its own — one trait of one predator moves behaviour almost as much as
  adding a second predator.
- **Attack delay** behaves sensibly: a longer pause means the agent survives longer and
  accumulates more hiding in absolute terms, while the rate barely moves. Opportunity, not
  urgency.
- **Maximum stamina is a genuine null** — flat on all three measures across its entire
  30–150 range. How long a predator *can* chase appears not to matter, presumably because
  encounters resolve long before stamina binds.
- **Bushes available** is the only factor that raises hiding *and* survival together. More
  cover is simply more usable cover.
- **Ambush predators** barely move hiding while cutting survival ~14%, consistent with the
  earlier finding that removing them entirely did not restore injured cover use.

Four of the eight declared predator traits are pinned to single values in this config and
cannot be analysed at all: move interval, stamina recovery, hunt threshold, lose-interest
multiplier.

## The finding worth arguing about

**The agent's behaviour is dominated by a trait it cannot perceive.**

Detection range is not in the observation. The agent sees a fixed class marker and a smell
drawn from an independent random stream; the predator's detection range, attack delay,
attack reach and stamina never reach its senses.

So this is not an agent recognising a dangerous-looking predator. It is an agent reacting to
*being detected and chased more often*, which is downstream of a property it has no channel
for. The effect is causally identified because the trait is redrawn at random each episode,
but the entire pathway runs through experienced consequence rather than perception.

That distinction matters for how the result gets described. "The agent hides more from
predators that can see further" is true and invites a perceptual reading that the environment
makes impossible.

## Method

Binomial GLM, bush steps out of episode length, standard errors scaled for overdispersion.
Trait models use the 333,743 episodes with exactly one active predator, so that the trait
value is unambiguous; scene-composition models use all 1,000,000.

Full coefficient tables are on the shareable page and in
`results/analysis/trajectory_glm/glm.json`.

### Two caveats

**Ignore the p-values.** At a million episodes every term is significant. Maximum stamina, at
p = 0.70, is the only one that failed — which indicates how little the test discriminates
here. Effect sizes are the only meaningful output.

**The variance model is wrong, though not fatally.** Overdispersion is 23–27× what a binomial
assumes, because hiding arrives in bouts and consecutive steps are not independent. Standard
errors are scaled accordingly; a beta-binomial or episode-clustered fit would be more honest.
The effects are far too large for this to change any conclusion.

## Follow-ups

1. Run the same analysis across all ten arms — the data is already collected — to ask whether
   the healing-rate parameter changes any of these sensitivities.
2. Split hiding by injured versus healthy segments within an episode. The standing open
   question is why an *injured* agent does not use cover, and that is a within-episode
   condition this episode-level analysis cannot address.
3. Consider whether "share of episode hiding" should be conditioned on survival at all, given
   how strongly detection range moves episode length.
