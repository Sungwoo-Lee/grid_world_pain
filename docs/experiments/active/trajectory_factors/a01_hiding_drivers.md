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
**Reviewed by:** `plan-reviewer`, 2026-08-25, two rounds; then corrected again after a
reader challenge — see [Corrections](#review-response). Finding 3 was rewritten after the
original claim ("the agent cannot perceive its own injury") was found to be **wrong**.
**Replicated across all 10 trained arms** — see [Cross-arm](#cross-arm).
Injury and pain in depth: [Injury-dependent hiding](#injury-deep). Comparing the ten agents: [Model effect](#model-effect). Other runs: [What else](#other-runs).
A step-level deep dive on the pain channel is in [Interoceptive pain](#nociception) —
which concludes that the interoceptive channel's specific role is **not identified** by this
data, and names the experiment that would settle it.

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
| Own injury | no *direct* readout | `injury_observable: false` removes the instantaneous value. |
| Pain (interoceptive nociception) | **observable, lagged** | A 12-step buffer of *injury levels* convolved with a kernel peaking 3 steps back. The agent **does** feel its wound — smoothed and delayed. The buffer is zeroed at reset, so a wound it wakes up with is unfelt on step 1 and fades in over ~10 steps (perceived value 0 → 28 → 58). |
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
not matter in a symmetric arena, and does not. **Predator stamina** is a trait with no route to
any of the agent's senses, and lands at exactly zero.

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

### 3. The agent responds to what pain *implies*, not to pain itself

It feels its wound. So the question is not whether injury reaches it, but what it does with it.
The answer depends entirely on **how the wound was acquired**, and the two cases point opposite
ways.

**A wound it woke up with makes it hide slightly *less*.** The environment assigns a random
injury, 0 to 100, at the start of every episode. Tracked step by step, the effect on hiding
appears exactly as the perception fades in — and it is negative:

| Step | perceived pain from a wound woken up with | hiding, lightly vs badly hurt |
|---|---|---|
| 1 | 0 (buffer empty) | +0.1 pp |
| 3 | 18 | +0.7 pp |
| 5 | 38 | -0.7 pp |
| 10 | 58 (saturated) | **-2.0 pp** |
| 20 | — | **-4.6 pp** |

Sign-stable across all ten trained agents (-0.94 to -1.28 pp per SD). It is small, and it
cannot be cleanly separated from differential survival, which also grows with elapsed time.

**A wound it earned makes it hide far *more*.** Conditioning on no predator within two tiles,
so that current proximity cannot be doing the work, hiding rises steeply and monotonically with
the pain signal the agent actually receives:

| Perceived pain | hiding, no predator within 2 tiles | hiding, predator near |
|---|---|---|
| none | 12.6% | 78.5% |
| 0-10 | 11.4% | 50.6% |
| 10-20 | 12.7% | 48.3% |
| 20-35 | 15.4% | 54.1% |
| 35-55 | 21.3% | 64.0% |
| 55+ | **31.0%** | 71.9% |

From 11.4% to 31.0% — nearly threefold — on the same perceptual channel that, when the wound
was randomly assigned, produced a small negative.

**The reconciliation.** The agent is not hiding reflexively because it hurts. It is treating
pain as *evidence that a predator is nearby*, which is a sound inference: in this world the
only way to get hurt is to be attacked. A wound it wakes up carrying is the one case where that
inference is false — nothing attacked it — and there the response is absent.

Two caveats. Earned pain is not randomised, so that table is an association: the agent feels
pain *because* it was attacked, and "no predator within two tiles right now" does not mean no
predator was near recently. And the randomised effect is entangled with survival. What the two
together do establish is that the pain channel alone does not drive hiding — its behavioural
meaning depends on the context that produced it.

For completeness, the raw injury-level counts are confounded in the same way, and conditioning
on circumstance makes them heterogeneous and reverses them where a predator is near:

| | mild injury (0-25) | severe injury (>=50) | difference |
|---|---|---|---|
| no predator near, no recent damage | 12.9% | 15.2% | +2.4 pp |
| no predator near, recent damage | 9.4% | 21.7% | +12.3 pp |
| predator near, no recent damage | 56.9% | 35.5% | **-21.5 pp** |
| predator near, recent damage | 36.9% | 40.2% | +3.3 pp |

The +12.3 pp cell — hurt recently, no predator visible now — is the pain-as-evidence response
in isolation.

### 4. Hunger looks like a cause and is largely a footprint

Raw counts are emphatic: 50.2% hiding at nutrition below 25, against 9.1% above 75. Read
straight, hunger drives hiding. Three pieces of evidence say the arrow mostly runs backwards.

- **Hiding blocks foraging.** In a bush the agent eats on **7.3%** of steps; outside, **25.7%**.
  Food is placed away from bushes at reset, though respawning food samples without an occupancy
  check and can land on one — which is the likely source of the residual 7.3%. Either way cover
  is bought with meals.
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

Outcome: fraction of the agent's steps spent in a bush. The spawn row is excluded from
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

**Round three — a reader caught a substantive error, and this one mattered.** The document
originally claimed the agent "cannot perceive its own injury", resting on the config flag
`injury_observable: false`. That flag only removes the *instantaneous* readout. The
interoceptive nociceptor turns out to convolve a buffer of **injury levels** — not damage
events — so the agent does perceive its wound, smoothed and lagged. Reading the flag and
stopping there was the error, and it was mine; the reviewer's round-one pass had corroborated
it, so two passes carried the same mistake. Finding 3 has been rewritten around the signal the
agent actually receives, and the conclusion is stronger and more interesting than the one it
replaces: the same pain channel drives hiding *up* threefold when the wound was earned, and
slightly *down* when the wound was randomly assigned — because pain in this world is evidence
of a predator, and a wound you woke up with carries no such evidence. The
"imperceptible variable returns null" control claim has been withdrawn as invalid.

Round two also produced two upgrades adopted above: the start-injury effect is *zero* at the
steps of maximal live contrast, with the small late-window negative attributable to
differential survival; and the injury-versus-nutrition per-step curves were read as a matched
control pair. That reading is **withdrawn**: round three showed injury is perceivable, so the
pair does not contrast a perceivable with an imperceptible variable. Its remaining moderate points on table subsets, reproduction
scope, and the wording of "dissolves" and "rules out" are all incorporated; the hardcoded
scent channels it flagged in the script are now derived from the run's config, with a hard
failure if a run's configuration does not separate the two classes.

<a id="nociception"></a>
## Interoceptive pain in depth

Finding 3 establishes that the agent feels its wound through a lagged, smoothed channel. This
section goes under that at step level. It is more negative than the first draft: an
adversarial review found a bug in the reconstruction and two overstated magnitudes, and the
corrected picture is that **this analysis cannot isolate the interoceptive channel's
contribution at all**. What it can do is bound it and say precisely what would settle it.

### The signal, and how it was reconstructed

The trajectory store records `injury_level` per step but not the observation vector, so the
perceived signal is reconstructed: a twelve-slot buffer of injury levels, zeroed at reset,
written once per step, convolved with a kernel that ignores the current step and peaks three
steps back. Perceived pain and actual injury therefore routinely disagree — a fresh wound is
not yet felt, a healed one still is.

Because the action producing row `t` was chosen on the row `t-1` observation, every table below
pairs hiding at `t` with the percept at `t-1`.

> **Reconstruction bug, found in review.** The first version of this section used a boundary
> guard that leaked the episode's random *starting* injury into the first few reconstructed
> steps — the environment never writes it into the buffer. About 5% of steps were inflated, by
> up to 13.8 points, concentrated exactly in the early-episode window. It fabricated an
> all-zero column that an earlier draft of this section interpreted as a finding. Fixed;
> everything below is post-fix. Recorded in the Known Bugs registry.

### Which signal does the behaviour follow?

Hiding by actual injury (rows) against perceived pain (columns), no predator within two tiles:

| actual injury | felt 0 | felt 0-8 | felt 8-18 | felt 18-32 | felt 32-50 | felt 50+ |
|---|---|---|---|---|---|---|
| **0** | 12.8% | 13.9% | 26.5% | 48.0% | — | — |
| **0-10** | 4.1% | 8.7% | 9.5% | 15.6% | 42.7% | — |
| **10-25** | 6.6% | 10.5% | 9.2% | 10.1% | 24.6% | 44.3% |
| **25-45** | 10.2% | 12.2% | 13.2% | 11.6% | 12.3% | 34.0% |
| **45-70** | 11.9% | 12.2% | 13.8% | 15.5% | 18.0% | 26.8% |
| **70+** | 11.9% | 14.6% | 16.6% | 19.7% | 26.3% | 28.2% |

Along a row, hiding rises with felt pain; down a column, actual injury adds little. But this is
**a consistency check, not a discovery**. The observation vector contains the nociceptor and no
injury readout, so behaviour *cannot* depend on actual injury except through the percept — the
architecture guarantees the asymmetry. What the table earns is confidence that the
reconstruction captures the right variable.

Two honest qualifications. The rise is not steep in every row: the middle rows are flat until
the highest felt-pain column. And the cells are not exchangeable — at fixed actual injury,
higher felt pain means injury was recently *higher*, i.e. the agent has been healing, and
healing requires resting, which co-occurs with being in a bush (20.3% in-bush when resting
versus 13.2% when not). Some of the along-row rise is that composition.

### Being struck cannot happen in cover

Across **10,779,288** damage steps in the run, the agent was in a bush on exactly **zero** of
them. This is not statistical; it is three mechanical guarantees in this world's configuration:
bushes block animals from entering, so a predator cannot reach a concealed agent; ambush
resources never respawn and are placed disjoint from bushes at reset; and rocks damage only on
overlap, which requires standing on a rock tile. A differently-configured world could break any
of the three, so this is a fact about this run rather than about the environment in general.

It also explains why exposure *rises* into a hit — being out of cover is how the hit happened.

### The response to being hurt is real but modest

Hiding aligned on damage events, measured against the pre-window baseline ten steps before:

| steps from hit | -10 | -1 | 0 | +2 | +4 | +6 | +12 | +24 |
|---|---|---|---|---|---|---|---|---|
| all events | 12.9% | 7.3% | **0.0%** | 11.5% | 13.8% | 12.4% | 12.9% | 13.1% |
| isolated events | 15.3% | 11.8% | **0.0%** | 18.9% | 19.8% | 19.9% | 18.3% | 17.0% |
| *all vs baseline* | 0.0 | -5.6 | -12.9 | -1.4 | **+0.9** | -0.5 | 0.0 | +0.2 |
| *isolated vs baseline* | 0.0 | -3.5 | -15.3 | +3.6 | +4.5 | **+4.6** | +3.0 | +1.7 |

An earlier draft quoted "+8 pp at peak, still elevated twenty-five steps on", measured against
the lag -1 baseline. That was wrong twice over. The lag -1 baseline is mechanically depressed
(you must be exposed to be hit). And the isolated-event filter — no further damage for
twenty-five steps — is a **collider**: hiding prevents damage, so the filter preferentially
keeps the stretches in which the agent did hide.

The defensible statement is a range. The post-hit increase is **+0.9 percentage points** on all
events and **+4.6** on the collider-selected subset, so somewhere between about one and four and
a half points, with the upper end inflated by selection. The all-events curve returns to
baseline within about six steps; only the selected subset stays elevated.

### Context changes what the same pain is worth

| felt pain | no predator within 2 tiles | predator within 2 tiles |
|---|---|---|
| none | 12.6% | 75.8% |
| 0-10 | 11.5% | 51.4% |
| 10-20 | 12.8% | 49.5% |
| 20-35 | 15.5% | 55.9% |
| 35-55 | 21.6% | 66.1% |
| 55+ | **32.7%** | 74.4% |

This is the most robust result in the section — it barely moved under the bug fix. With no
predator in sight, high felt pain accompanies nearly three times the hiding of low felt pain.
With a predator present the agent is already hiding on half to three-quarters of steps and the
gradient is proportionally much smaller: the direct cue dominates.

### Rising versus falling pain: no clear asymmetry

| felt pain | falling | rising |
|---|---|---|
| 8-18 | 12.5% | 13.6% |
| 18-32 | 15.0% | 15.1% |
| 32-50 | 22.4% | 19.6% |
| 50+ | 32.2% | 30.5% |

An earlier draft read a consistent falling-above-rising gap as evidence of reverse causation.
After the one-step pairing correction the gap is inconsistent — rising is higher at 8-18, equal
at 18-32, lower above. This test does not adjudicate anything and is reported only to close it
out. Even the original gap had at least two explanations besides reverse causation, including
simple response latency: after a hit, pain is still rising while the agent is en route to
cover and falling once it has arrived.

### The alternative this analysis cannot rule out

Everything above is consistent with the agent making no use of the interoceptive channel at
all. The policy is **recurrent** — a memory network can carry "I was attacked" forward
indefinitely with no pain input whatsoever. And the agent has a separate **exteroceptive**
nociception channel that fires instantly on contact. So the post-hit rise, its timing, and the
internal-state dependence are all equally consistent with: felt the contact, remembered it, went to
cover — with the smoothed interoceptive trace playing no part.

Both readings predict every table in this section. The one causal handle available — the
randomised starting injury, which loads the interoceptive channel *without* any attack — moves
hiding slightly **negative**, which fits the deflationary reading at least as well.

**So the honest verdict is that the specific contribution of interoceptive pain is not
identified here.** What is established: hiding is systematically coupled to the pain state,
that coupling is much stronger when no predator is directly perceivable, cover is mechanically
protective, and the coupling holds in all ten trained agents.

Settling it requires an intervention on the pain channel that does not also imply a predator:
replay the trained agent with the interoceptive input clamped to a constant, or set to a
falsified value, and see whether the behaviour changes. The collection pipeline could support
that; this analysis could not do it. That is the recommended next experiment.

<a id="cross-arm"></a>
## Replication across all ten agents

The same analysis was run on all ten arms of the resting-bonus sweep. Because every arm replays
the **identical** million world draws, this is a perfectly paired comparison of agents: any
difference is the agent, never the environment.

Twelve of the fourteen randomised factors keep their sign in all ten independently trained
agents. The only two that flip are `pred_max_stamina` and `n_rocks` — the two factors this
analysis identifies as null, which is exactly the behaviour a null should show.

| Factor | a01 | range across the other nine |
|---|---|---|
| predator detection range | 12.51 | 11.00 to 12.04 |
| **rabbit's scent** | **3.30** | **4.06 to 5.20** (stronger everywhere else) |
| spawn distance to bush | -3.62 | -3.72 to -5.03 |
| starting nutrition | 3.54 | 3.39 to 4.61 |
| predator's scent | 2.02 | 2.17 to 3.65 |
| starting injury | -1.02 | -0.94 to -1.28 |
| predator max stamina | 0.08 | -0.03 to 0.17 (**sign flips**) |
| number of rocks | 0.02 | -0.15 to 0.18 (**sign flips**) |

The scent false alarm is not a quirk of one training run — it is present in every agent, and
a01, the arm this document analyses, has the **weakest** version of it.

The ten arms differ in exactly one setting: `body.recovery_accel_rate`, the extra healing
earned for *continuing* to rest. The values are **0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0,
2.7** — denser at the low end, not a linear ramp. (An earlier version of the cross-arm table
labelled these with an assumed linear ramp, which was wrong for seven of the ten arms. The
labels were cosmetic — the parameter is never a regressor — but the script now reads each value
from the run's own config rather than assuming.)

Overall hiding ranges 16.6% to 19.0% and mean survival 180.3 to 189.9 steps across arms. No
causal claim is made about the resting-bonus parameter: one training run per arm cannot
separate the parameter from the run.

<a id="injury-deep"></a>
## Injury-dependent hiding: pain state versus attack memory

The pain section closed on an unresolved question — is the agent using its pain signal, or just
remembering that it was attacked? Two tests narrow it.

### Same felt pain, different origin

Cross-tabulating hiding by felt pain against the number of hits taken earlier in the episode
(no predator within two tiles) separates the two. Prior-hit count is a memory variable that
cannot be recovered from the current pain value.

| felt pain | 0 prior hits | 1 | 2 | 3-4 | 5+ |
|---|---|---|---|---|---|
| felt 0 | 23.9% | 18.2% | 13.8% | 11.3% | 9.5% |
| 8-18 | 14.6% | 15.3% | 14.3% | 13.4% | 11.0% |
| 32-50 | 19.1% | 26.6% | 24.3% | 24.0% | 17.4% |
| **50+** | **22.6%** | **40.9%** | 40.0% | 40.7% | 29.1% |

Read the bottom row. At high felt pain, an agent that has actually been hit once hides
**40.9%** of the time. An agent with *the same felt pain* but **zero hits** — which can only
happen when the pain comes from the randomly assigned starting wound — hides **22.6%**. Nearly
half as much, from an identical interoceptive signal.

So the pain value alone does not determine the response; how the pain was acquired does. This is
the sharpest evidence available here for the "pain as evidence of a predator" reading, and the
zero-hit column is randomised, which is what gives it force.

Two caveats. The zero-hit high-pain cells are necessarily early in the episode (a starting wound
heals within ~20 steps), so time is confounded with origin. And beyond one hit, more hits means
*less* hiding — an episode with five hits is one where the agent kept failing to reach cover, so
hit count also marks incompetence or an inescapable situation, not just memory.

### Same felt pain, different age

| felt pain | 1 step after a hit | 2-3 | 4-6 | 7-11 | 12+/never |
|---|---|---|---|---|---|
| 18-32 | 8.6% | 15.1% | 15.9% | 14.1% | 15.6% |
| 50+ | 14.8% | 28.8% | 35.3% | **38.7%** | 27.2% |

Immediately after a hit, hiding is at its lowest — the agent is out in the open, which is how it
was hit — and it climbs over the following steps, peaking seven to eleven steps later. The same
felt pain produces very different behaviour depending on how recently it arrived.

### Approaching death

| injury | 0 | 0-25 | 25-50 | 50-75 | 75-90 | 90-100 |
|---|---|---|---|---|---|---|
| hiding | 13.6% | 10.3% | 15.0% | 19.3% | **20.7%** | 17.6% |

Hiding rises with injury up to the 75-90 band and then **falls** as injury approaches the lethal
ceiling of 100. Whatever the agent does near death, it is not more cover-seeking.

### The same relationship in all ten models

| arm | felt 0-8 | 8-18 | 18-32 | 32-50 | 50+ | slope |
|---|---|---|---|---|---|---|
| a01 (premium 0) | 14.1% | 15.7% | 19.1% | 28.3% | 41.6% | +27.5 |
| a07 (premium 1.2) | 13.7% | 16.3% | 18.5% | 21.3% | 33.4% | +19.7 |
| a10 (premium 2.7) | 13.2% | 16.0% | 19.1% | 22.5% | 37.0% | +23.8 |

Every one of the ten independently trained agents shows the same shape — a shallow dip at low
pain, then a steep climb — with slopes from +19.7 to +27.5 percentage points. The pain-hiding
coupling is a property of the task, not of any one training run. The slope is weakly smaller in
agents with a larger resting bonus (Spearman rho = -0.62, p = 0.054), which is suggestive only
and carries the same attribution problem as everything else in the cross-arm comparison.

<a id="model-effect"></a>
## Comparing the ten models against each other

All ten arms replay the **identical** million world draws, and both the policy and the
environment are deterministic given (world, agent). So the outcome is a complete
1,000,000 x 10 table with **no noise term at all**, and its variance decomposes exactly into
three parts: which world was drawn, which model was used, and their interaction.

| | which world | **which model** | model x world |
|---|---|---|---|
| bush-dwell fraction | 57.8% | **0.14%** | 42.1% |
| survival steps | 83.0% | **0.02%** | 17.0% |

**The model you pick barely matters on average, and matters a great deal case by case.** The
ten agents' overall hiding rates span 3.2 percentage points where the world moves it by 25; the
across-agent spread in survival is 9.6 steps where the world moves it by 187. Yet the
interaction term is 300x the model main effect for hiding. Averaged over a million worlds the
ten models look nearly the same; dropped into any *particular* world they diverge sharply.

Because the worlds are paired, the small average differences are measured extremely precisely —
confidence intervals of about +/-0.06 pp. Against a01, the zero-premium anchor:

| arm | premium | dwell vs a01 | survival vs a01 |
|---|---|---|---|
| a02 | 0.3 | +2.81 pp | -1.97 steps |
| a05 | 0.9 | +1.43 pp | -5.86 steps |
| a07 | 1.2 | +3.16 pp | -7.56 steps |
| a08 | 1.5 | -0.02 pp | **-9.65 steps** |
| a10 | 2.7 | +3.10 pp | -6.73 steps |

The ordering of agents is stable across environment conditions (rank correlations +0.83 to
+0.99 between the overall ordering and the ordering within 0-, 1- and 2-predator episodes), so
these differences are systematic rather than arbitrary.

### Is the resting bonus responsible? Not attributable.

Survival falls monotonically as the resting bonus rises: Spearman rho = **-0.818**, p = 0.004
across the ten levels. That looks like a dose-response. Four checks say it cannot be claimed.

1. **Training length is matched** — all ten reached 100,000,000 steps to within 79, with 1000
   checkpoints each. This confound is genuinely excluded.
2. **The mechanism fails.** The natural story is that a bigger resting bonus makes the agent
   rest more, so it forages less and starves. Eating does fall (rho = -0.65, p = 0.04) and
   starvation does rise, but the **first link is absent**: resting shows no trend with the bonus
   at all (rho = +0.25, p = 0.49). The chain's opening move does not happen.
3. **One run per level, and a single seed.** Every arm trained from seed 42. A shared seed fixes
   the initialisation but not the outcome: once the parameter changes the dynamics the learning
   trajectories diverge immediately. Each level is therefore still one draw from the
   distribution of training outcomes at that parameter, and one draw cannot separate the
   parameter's effect from where that particular run happened to land.
4. **Node is collinear with the parameter.** The arms were assigned to compute nodes in pairs
   following the healing order, so node index correlates with the bonus at rho = +0.886 — and
   node predicts survival *slightly better* (rho = -0.935) than the bonus does (-0.818). All
   five nodes carry the same GPU model, so there is no obvious mechanism, and five nodes across
   ten runs is far too few to test one. But the design cannot tell the two apart.

**Verdict: the ten models differ measurably and systematically, and survival tracks the resting
bonus, but this design cannot attribute that trend to the bonus.**

### What would settle it

Replicate seeds, not more parameter levels. Three seeds at four levels (12 runs) is strictly
more informative than one seed at ten levels (10 runs), because it measures the run-to-run
variance directly and gives the trend an error bar. A sweep of the current shape can only ever
produce a suggestive line.

Two cheap additions to the launch procedure: **randomise the node assignment** across parameter
values rather than allocating in order, and **vary the training seed** across replicates. The
repository currently holds no same-config multi-seed replicates for this family, so there is no
way to calibrate run-to-run noise after the fact.

Produced by `scripts/analysis/supplementary/modeleffect.py`.

<a id="other-runs"></a>
## What else could be analysed

This document covers **one family of runs**: `rppo_restprem`, final checkpoint (~100M steps),
1,000,000 episodes replayed per arm. The repository holds 299 recurrent-PPO runs, 113
Dreamer-SRL and 36 DreamerV3. The families worth considering next:

| family | runs | trained to | config format | why it is interesting |
|---|---|---|---|---|
| **`rppo_restpremNH`** | 10 | ~59M | modern | **The paired counterfactual.** Identical to `restprem` — same ten healing values, same world — except the **ambush predators are removed entirely**. Directly tests whether the hiding behaviour depends on cover also being dangerous. |
| `rppo_bushrefuge` | 4 | 100M | modern | Fully trained; cover-focused design. |
| `rppo_nmn_g32_b04` | 3 | 36M | modern | Neuromodulation (FiLM) architecture. |
| `rppo_nmn_tempceil10_p3_film_g1` | 3 | 10M | **legacy** | The only family with **`perceptual_noise.enabled: True`** — injury degrades the senses, giving the agent a second, indirect route to knowing it is hurt. The natural contrast for the unresolved pain question. |
| `rppo_hg02` / `rppo_hg04` | 3 + 3 | 10M | modern | Hypervigilance series. |
| `interoNocicept_*` | 4 | 100k | legacy | Too short to be worth replaying. |

Two practical notes.

**Budget matching.** `restpremNH` trained to ~59M steps against `restprem`'s ~100M, so a naive
comparison confounds the manipulation with training length. `restprem` kept 1000 checkpoints, so
a step-matched checkpoint near 59M exists and should be used instead of the final one.

**Legacy configs.** `scripts/analysis/hiding_drivers.py` derives its slot layout from
`environment.entities`. The `nmn_tempceil10` and `interoNocicept` families predate that key and
carry `predators:` / `neutral_animals:` instead; the tool would need a legacy branch in
`slot_layout` before it could read them.

## Data

- 1,000,000 episodes, 189,906,610 steps, mean survival 189.9 steps, overall hiding 16.62%.
- Deaths: 42.5% killed, 30.9% starved, 26.7% survived the 500-step cap.
