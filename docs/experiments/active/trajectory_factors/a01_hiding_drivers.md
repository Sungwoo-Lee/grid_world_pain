# What makes this agent hide? A million-episode factor analysis

**Shareable page:** https://claude.ai/code/artifact/1351009f-d7f7-4114-a290-f6582bb9a004
**Reproduce (ranking, multivariate models, same-step cross-tabs):**
`$CONDA/bin/python scripts/analysis/hiding_drivers.py --run results/JAX_RecurrentPPO/20260810-185749_rppo_restprem_a01_n106`
where `$CONDA` is `/home/vncuser/miniconda3/envs/grid_world_pain`. The supplementary passes
behind findings 2-4 — lagged proximity, the scent ladders, the targeted false-alarm split, the
injury conditional table, the injury window, and the eat-block check — are archived in
`scripts/analysis/supplementary/` and are *not* produced by the command above.
**Extends:** [[a01_factor_analysis]] (same run, same store, narrower question)
**Pipeline:** [[TRAJECTORY_COLLECTION_PIPELINE]]
**Reviewed by:** `plan-reviewer`, 2026-08-25, two rounds — final verdict *conclusion supported*.
It found one factual error in a supporting sentence; that is corrected and documented in the
[Review response](#review-response).

## Question

We took one trained agent, put it back into the exact world it had learned in, and let it live
one million separate lives. We recorded every step of every life — where it stood, what it
could smell, how hurt and how hungry it was, whether it was tucked inside a bush. Then we
asked one question of that record: **what makes this agent hide?**

The world rerolls itself at the start of every life. It picks how many predators appear,
how far each one can see, how hard and how quickly it strikes, how many bushes and rocks
exist, how much food is scattered about — and it also picks how *injured* and how *hungry*
the agent starts out. All of these are dice rolls made before the agent does anything, so
their effects are **causally identified**: this is a randomised experiment that the
environment runs on itself, a million times over.

The agent studied here is the arm of the resting-bonus sweep where resting earns no extra
healing (the "no-premium" anchor). One agent, one training run.

## What the agent can and cannot sense

This governs every result below, so it comes first. From the run's own saved settings:

| Sense | Setting | Consequence |
|---|---|---|
| Sight | range **0** | It sees only the tile it stands on. Blind at distance. |
| Smell | enabled, radius 20 | Its **only** distal sense. Covers the whole 10x10 map. |
| Own injury | **not observable** | No channel carries its wound level. |
| Pain | observable | A fading trace of *recent damage events*, not of a standing wound. |
| Hunger | observable | Via satiation, which tracks nutrition one-for-one. |

The two animal types were built to smell alike: a predator's scent is drawn around (0.7, 0.5)
on two odour channels, a harmless rabbit's around (0.5, 0.7), each with a spread of 0.3 and
clipped to [0,1]. They overlap heavily. Predators and rabbits *look* different — but the agent
cannot see them.

## Headline finding

**The agent responds to a smell it cannot disambiguate, and the mistake is measurably fatal.**

When a *predator* is randomly drawn smelling strongly predator-like, the agent hides more and
lives **30% longer** (89 to 115 steps). When a *harmless rabbit* is drawn smelling the same
way, the agent hides just as much more and lives **9% shorter** (193 to 176 steps).

Same behaviour, opposite consequence. And the extra hiding is aimed: it concentrates in
precisely the moments that misleading rabbit is nearby.

## The ranking

Every randomised factor, by how much it moves hiding. "pp" = percentage points of the episode
spent in a bush, per one standard deviation of the factor, so quantities on different natural
scales can be compared.

| Factor | Effect on hiding | Direction |
|---|---|---|
| number of predators (0/1/2) | **+16.0 pp** | more predators, far more hiding |
| predator's detection range (1-7) | **+12.5 pp** | keener predator, far more hiding |
| distance to nearest bush at spawn | -3.1 pp | further from cover, less hiding |
| starting nutrition (0-100) | +2.9 pp | better fed, more hiding |
| number of bushes (4-10) | +2.8 pp | more cover, more hiding |
| number of rabbits (0/1/2) | +2.4 pp | harmless animals still trigger it |
| predator's scent (predator-likeness) | +1.7 pp | smells more like a predator |
| **rabbit's scent (predator-likeness)** | **+1.6 pp** | a harmless animal smelling wrong |
| predator's attack delay | +1.2 pp | slower striker, slightly more hiding |
| number of food items (1-4) | -1.2 pp | more food, less hiding |
| **starting injury (0-100)** | **-0.8 pp** | more hurt, very slightly *less* hiding |
| predator's attack reach | -0.8 pp | |
| spawn distance to nearest food | +0.6 pp | further from food, slightly more hiding |
| number of ambush predators (2-12) | +0.2 pp | negligible |
| distance from map centre at spawn | +0.0 pp | negligible — **null control** |
| number of rocks (6-12) | +0.1 pp | negligible — **null control** |
| predator's max stamina (30-150) | -0.0 pp | no effect at all |

This is exhaustive: every quantity the environment randomises appears above. Three of them
double as controls, and all three behave as they should. **Rocks** are scattered like bushes but
conceal nothing, and unlike bushes they *hurt* (1-5 damage, high pain intensity). Their count
moves the behaviour not at all, so the agent is neither reacting to clutter nor treating a
damaging object as a reason to seek cover. **Where on the map the agent wakes up** should
not matter in a symmetric arena, and does not. **Starting injury** is a quantity the agent
provably cannot perceive, and lands at approximately zero.

One factor separates survival from strategy cleanly: spawning far from food cuts survival hard
(221 down to 164 steps) while barely touching hiding (15.2% to 17.5%). Food distance decides
whether the agent lives, not how it behaves.

## Four findings

### 1. Proximity dominates, moment to moment

Using each animal's position on the *previous* step to predict hiding on the current one, so
the agent's own choice cannot manufacture the correlation:

| Situation one step earlier | Share of steps spent in a bush |
|---|---|
| a predator within 2 tiles | **67.9%** |
| a rabbit within 2 tiles (no predator) | 17.9% |
| neither | 11.6% |

The lagged version is *stronger* than the same-step version (67.9% vs 63.1%), which argues
against the objection that predators merely linger near an already-hidden agent — though
consecutive steps are autocorrelated, so it weakens that objection rather than eliminating it. These are
associations, not causal effects — proximity is partly the agent's own doing.

### 2. The scent false alarm, and its price

Both animals' scents are randomised per episode, so these are causal ladders. Each table is
restricted to episodes containing **exactly one** animal of that type, so the scent is
unambiguous; rows are the extreme sixths of the predator-likeness range (below -0.2, and above
0.6). "Predator-likeness" is the difference between the two odour channels that separate the
classes, derived from the run's config rather than assumed.

| Rabbit's scent | Hides | Food per step | Starved | Killed | Survived |
|---|---|---|---|---|---|
| least predator-like | 14.9% | 0.227 | 27.6% | 45.0% | **193.3 steps** |
| most predator-like | 22.6% | 0.212 | 38.5% | 38.2% | **175.6 steps** |

| Predator's scent | Hides | Food per step | Starved | Killed | Survived |
|---|---|---|---|---|---|
| least predator-like | 29.7% | 0.197 | 24.6% | 71.6% | **88.9 steps** |
| most predator-like | 35.0% | 0.183 | 40.7% | 52.9% | **115.4 steps** |

The chain is visible end to end: misread the scent, hide more, eat less, starve sooner. Hiding
buys real protection — death-by-predator falls in both tables — but when the threat was never
real, the protection is worthless and the lost meals are not.

**Is this a targeted false alarm or just raised general vigilance?** A reviewer asked, and the
data answer. Holding the episode to exactly one predator and one rabbit, and splitting by the
rabbit's randomised scent:

| Extra hiding when the rabbit smells predator-like | |
|---|---|
| in moments when nothing is near | +8.0 pp |
| in moments when the predator is near | +6.4 pp |
| **in moments when that rabbit is near** | **+23.1 pp** |

The response is aimed at the rabbit, roughly three times more strongly than anywhere else.
Hiding while that rabbit is nearby rises from 26.7% to 49.8% — approaching how the agent
treats an actual predator (56.4%). This is a sensory misidentification, not a diffuse mood.

*Caveat.* Proximity here is same-step, and "rabbit is nearby" is itself partly the agent's
doing, so splitting on it conditions on a post-treatment variable. The three-way split is
therefore a *descriptive localisation* of where the extra hiding lands — which is exactly what
distinguishes an aimed response from a diffuse one — not a causal decomposition. The
"nothing near" row (+8.0 pp) is the diffuse-vigilance baseline, so the aimed component is the
+15 pp excess over it. The selection is small and works against the finding: when the rabbit smells
predator-like the agent spends slightly *less* time near it (13.0% of steps versus 14.3%),
so the surviving near-moments are the ones it could not avoid.

### 3. Injury does not drive hiding

The environment hands the agent a random injury, 0 to 100, at the start of every episode. If a
wounded animal were intrinsically more cautious, this is where it would show. Hiding instead
drifts gently *down*, 17.7% to 15.4%, as the starting wound worsens.

Wounds heal at 5 points per step, so a randomised starting injury is gone within about twenty
steps. Measured **only inside that window**, where the manipulation is genuinely live and the
gap in experienced injury is fivefold (mean 70.5 versus 13.3), and computed without
conditioning on survival:

| Starting injury | Hiding, steps 1-10 | Mean injury over that window |
|---|---|---|
| 0-20 | 20.2% | 13.3 |
| 80-100 | 19.2% | 70.5 |

A **-1.0 pp** effect — roughly 25 times smaller than detection range, and pointing the wrong
way. Broken down step by step it is *zero* at the steps of maximal live contrast (t=1-4,
where the injury gap is widest) and only becomes negative later in the window, which is the
signature of differential survival rather than of injury perception: lightly-hurt episodes that
got attacked early live on into the late window and hide, while their badly-hurt counterparts
are already dead. The reason sits in the sense table: **injury is not observable**. The agent has no
channel for it and feels pain only when actually struck.

Raw counts appear to say the opposite: 26.3% hiding at severe injury against 17.8% at none.
That comparison is confounded. Conditioning on what was actually happening makes it
heterogeneous rather than uniform, and reverses it outright where a predator is near:

| | mild injury (0-25) | severe injury (>=50) | difference |
|---|---|---|---|
| no predator near, no recent damage | 12.9% | 15.2% | +2.4 pp |
| no predator near, recent damage | 9.4% | 21.7% | +12.3 pp |
| predator near, no recent damage | 56.9% | 35.5% | **-21.5 pp** |
| predator near, recent damage | 36.9% | 40.2% | +3.3 pp |

So injury is largely standing in for "a predator was around", which is the factor doing the
work. One cell is worth naming rather than averaging away: with no predator near but damage
taken recently, badly-hurt states carry +12.3 pp more hiding. That is what the sense table
predicts — the agent cannot feel the wound, but it can feel the *strike*, and the pain trace
persists after the predator has gone.

### 4. Hunger looks like a cause and is largely a footprint

Raw counts are emphatic: 50.2% hiding at nutrition below 25, against 9.1% above 75. Read
straight, hunger drives hiding. Three pieces of evidence say the arrow mostly runs backwards.

- **Hiding blocks foraging.** In a bush the agent eats on **7.3%** of steps; outside, **25.7%**.
  A bush holds no food, so cover is bought with meals.
- **Randomised hunger acts immediately, in the *opposite* direction.** Binned by the
  randomised starting value and measured unconditionally over the first ten steps, agents that
  start starving hide **less**, not more: 16.4% at starting nutrition below 25, rising to 21.4%
  above 75. A hungry agent leaves cover to forage, from the first few steps. That is the exact
  reverse of the observational gradient.
- **Randomised starting nutrition points the other way** over the whole episode: better-fed
  agents hide *more* (11.6% to 20.0%), because only a fed animal can afford cover.

So the randomised evidence and the observational association point in opposite directions.
Hunger *is* perceivable — satiation tracks nutrition one-for-one — and the randomised effect
shows a real hunger-to-behaviour channel: it pushes the agent **out** of cover, which is
adaptive. The large positive observational association must therefore come from somewhere
else. Two mechanisms are in play and this design cannot separate their shares: **reverse
causation** (hiding blocks eating, so long bush stays produce low nutrition) and **context
confounding** (a predator nearby both pins the agent in a bush and prevents it feeding, so
predator presence drives hiding and hunger jointly).

## How to read these numbers

- **Randomised factors are causal** — the environment rolls them before the agent acts.
- **Everything else is association** — proximity, injury sustained, nutrition spent, damage
  taken are all partly the agent's own doing.
- **Every coefficient is causal for a composite**, "fraction of survived steps spent hidden",
  not for a per-step hiding propensity, because episode length is itself an outcome.
- **Ignore the p-values.** At a million episodes everything is significant. Ranking is by
  effect size throughout.
- **Randomised bodily state is transient** (injury gone in ~20 steps, nutrition depleting at
  1/step) while predator count persists all episode. Comparing their whole-episode effects
  partly compares persistence; finding 3 therefore reports the windowed estimate as well.
- **One run, one seed.** Nothing here compares training arms.

## Method

Outcome: fraction of the agent's chosen steps spent in a bush. The spawn row is excluded from
the outcome (the agent did not choose where it woke) and used only as a starting condition.
Model: quasi-binomial GLM on the rate, standard errors scaled by the Pearson overdispersion
(13-25 depending on model). Effects converted to percentage points at the observed mean via a
delta-method linearisation — accurate for the small effects tabulated, but not to be applied
to the large endogenous exposure coefficients.

**Truncation check.** A factor that shortens episodes could inflate its own apparent effect,
because late steps carry lower hiding. Recomputing on a fixed early window makes every effect
*larger*, so truncation cannot be generating them. For the rabbit-scent effect the share of
episodes reaching step 25 is flat across scent levels (67.1%-68.5%), so that check is unbiased.
For starting nutrition it is not (32.6% to 78.0%), so the whole-episode estimate is quoted
instead; for starting injury it varies mildly (71.5% to 61.2%), so finding 3 uses the
unconditional windowed rate.

<a id="review-response"></a>
## Review response

`plan-reviewer` verified the slot layout against the raw store, confirmed in `src/` that the
scent draws are sampled at reset and feed only the olfactory sensor, and independently refit
the rabbit-versus-predator scent contrast (z = 13.8). Its two critical findings and their
resolution:

1. *"The temporal-confound explanation for the hunger gradient is refuted by your own
   diagnostic — the gradient persists inside every time bin."* Correct, and this write-up never
   makes that claim; finding 4 attributes the gradient to reverse causation, with the
   time-controlled table as supporting evidence and the residual channel acknowledged.
2. *"The +40 pp rabbit-proximity figure is a conditional coefficient among endogenous,
   negatively correlated exposure shares, not an aggregation-bias counterpart of the
   step-level +6 pp; the univariate association is actually negative."* Correct. That figure
   is not used anywhere in this document; finding 1 reports only the lagged step-level
   cross-tab, labelled associational.

Its moderate findings on transient manipulation (finding 3, now windowed), the circular
alignment assertion (a per-shard contiguity check is in `scripts/analysis/hiding_drivers.py`),
and the false-alarm versus redundant-cue ambiguity (tested directly in finding 2) were all
addressed, and a second round was requested on the revised text.

**Round two found one factual error, and it was mine.** An earlier version of finding 4 claimed
that randomised hunger "does nothing immediately — 18-20% flat across every hunger level" in
the first ten steps. That was false. The 18-20% figures came from a table binned by the
agent's *contemporaneous* nutrition at each step, which is not the randomised draw; I then
described them as though they were. Binned by the actual starting value and measured
unconditionally, hiding runs 16.4% to 21.4% — a real gradient, and one that points **opposite**
to the observational association. Correcting the error strengthened the finding rather than
weakening it, and finding 4 is now written around the corrected numbers. The offending script
(`scripts/analysis/supplementary/timectrl.py`) is archived with its wart documented so the
same misreading cannot be repeated.

Round two also produced two upgrades adopted above: the start-injury effect is *zero* at the
steps of maximal live contrast, with the small late-window negative attributable to
differential survival; and the injury-versus-nutrition per-step curves form a matched control
pair — the variable the agent cannot perceive does nothing immediately, the one it can perceive
acts within about three steps. Its remaining moderate points on table subsets, reproduction
scope, and the wording of "dissolves" and "rules out" are all incorporated; the hardcoded
scent channels it flagged in the script are now derived from the run's config, with a hard
failure if a run's configuration does not separate the two classes.

## Data

- 1,000,000 episodes, 189,906,610 chosen steps, mean survival 189.9 steps, overall hiding 16.62%.
- Deaths: 42.5% killed, 30.9% starved, 26.7% survived the 500-step cap.
