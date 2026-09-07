---
title: "Does the agent hide more or less when injured? Reconciling two opposite claims"
topic: diagnosis
status: active
created: 2026-09-03
last_updated: 2026-09-03
---

# Does the agent hide more or less when injured?

## Question

Two live claims in this project point opposite ways about the same-sounding behaviour, and both
would end up in a paper if nobody checked.

The **sensor-ladder study** — fourteen agents that differ only in what they can sense, each replayed
through a million test worlds — reports that an agent handed a bigger wound at the start of an
episode **spends more of its next twenty-five steps standing in a bush**, in thirteen of the
fourteen agents.

A note carried forward in the project's daily log on 1 September says the opposite: that **"hide
more when injured" is the target behaviour and is currently the OPPOSITE of what the agent does**,
and that two attempts to reverse it have already failed. That note is not itself a measurement; it
is a one-line summary of a separate line of work — the *bush-refuge* studies — in which an injured
agent uses cover **less** than an uninjured one.

**Verdict: they are not in conflict, and they are not even about the same measurement.** They are
about different trained agents, in a different world, evaluated in a different place, over a
different stretch of the episode. More usefully, this analysis found the single behaviour that sits
underneath both: **a wounded agent stops moving and rests.** Whether that shows up on the scoreboard
as "more hiding" or "less hiding" depends entirely on how far the nearest bush happens to be when it
freezes. Both claims are true statements about that one behaviour, seen from two different vantage
points.

This is a **post-hoc** analysis. No predictions were registered in advance, and no training was run.

## Verdict in one table

| Dimension | Sensor-ladder claim ("hides more") | Bush-refuge claim ("hides less") | Same? |
|---|---|---|---|
| Learning algorithm and network | Recurrent PPO, GRU, 128 units, seed 42, **neuromodulator off** | identical in every listed respect | **yes** |
| Which trained agents | fourteen `lad_*` runs, Aug 2026 | twenty `restprem` / `restprem`-no-ambush runs, Aug 2026 | no — different runs |
| What the agent can sense | fourteen different settings | one setting, functionally identical to the ladder's poorest arm | partly |
| Bush behaviour in the training world | hides the agent from sight only | hides it **and** physically blocks predators from entering | **no** |
| Where behaviour was measured | in the training world itself | in a stripped-down test scene the agent never trained in | **no** |
| Is a predator present when measured | in two-thirds of episodes | never, in the condition that carries the claim | **no** |
| Is there food to compete for | yes | none at all; hunger is pinned full | **no** |
| Starting wound | drawn at random, 0 to 100 | fixed at exactly 0 or exactly 70 | no |
| Stretch of episode measured | first 25 steps | all 101 steps | **no** |

Four of those differences are each, on its own, enough to change the sign of the answer. The two
claims never met.

## 1. Are these the same agents?

**Same agent class; different training runs. The prior that the second claim concerns the
neuromodulated agents is refuted.**

Read from each run's own saved settings file — the copy the trainer wrote when the run started, not
a fresh re-resolution of the source configs:

| | fourteen sensor-ladder runs | ten rest-premium runs | ten no-ambush replicate runs |
|---|---|---|---|
| algorithm | Recurrent PPO | Recurrent PPO | Recurrent PPO |
| recurrent cell / width | GRU / 128 | GRU / 128 | GRU / 128 |
| **neuromodulator** | **off** | **off** | **off** |
| training seed | 42 | 42 | 42 |
| learning rates, discount, clip, GAE, return mode | identical across all three families | | |

So both claims are about the project's plain, unmodulated baseline agent. Neither is about a
modulated run. Whatever separates them, it is not the modulator.

**A detail that turns out to matter a lot.** The rest-premium agents' sensory setup is *functionally
identical* to the sensor ladder's `A_baseline` arm — the poorest-sensed of the fourteen. Both have a
one-cell visual field (they see only the square they stand on), one omnidirectional whiff of smell
covering the whole map, no direct readout of their own injury, and the same twelve-step smoothed
internal wound signal. The rest-premium settings file predates several sensory options and simply
omits them; every omitted key takes the value the ladder's `A_baseline` arm sets explicitly (blur
off, occlusion off, smell grid range 0).

`A_baseline` is the **one** ladder arm out of fourteen whose starting-wound effect is negative
(−2.3 percentage points). The sensory twin of the "hides less" agents is the single ladder arm that
also says "hides less". That is not a coincidence; §4 explains it.

## 2. Are they the same environment?

**No — and there are two separate environment differences, one in training and a much larger one in
where behaviour was measured.**

### 2.1 The training worlds differ in one mechanic

Both are 10×10 worlds with the same creatures, the same counts, the same pouncing predator and the
same 500-step cap. They differ in what a bush *is*:

- **Sensor ladder:** a bush conceals the agent. Animals may still walk onto it.
- **Bush-refuge line:** a bush conceals the agent **and** blocks animals from entering it. It is a
  physical safe room.

In the bush-refuge world the agent was never once damaged while in a bush across 10.8 million
recorded damage steps. Cover there is absolute; in the ladder it is not.

The two lines also differ in how healing is priced — the ten rest-premium arms sweep how much extra
a *continuous* rest streak heals, from no bonus at all up to a factor of about 130,000, while the
ladder uses the project default. The sweep was deliberately built so that all arms shed a wound of
70 in about the same 13–15 resting steps, so the length of the wounded window is matched.

### 2.2 The measurement environments differ enormously

This is the bigger difference, and it is the one most likely to be missed, because both are called
"evaluation".

**The ladder measures behaviour in the training world.** A million fresh worlds drawn from exactly
the distribution the agent trained on: 1–4 food items, 0–2 hunting predators, 0–2 harmless rabbits,
2–12 stationary ambush hazards, 4–10 bushes scattered at random, a random starting position, a
random starting hunger and a random starting wound.

**The bush-refuge claim measures behaviour in a purpose-built test scene the agent never trained
in.** In the condition that carries the claim:

| | value |
|---|---|
| episode length | 101 steps |
| food | **none** |
| hunger | pinned at full; it never depletes as a motive |
| hunting predators | **none** |
| animals present | one harmless rabbit, wandering |
| bushes | exactly **one**, at a fixed square |
| agent start | a fixed square, **three cells from that one bush** |
| starting wound | exactly 0, or exactly 70 |

So: no threat to hide from, no food to leave cover for, a single patch of cover the agent must walk
three squares to reach, and only two wound levels. The claim compares the wounded run against the
unwounded one in that scene.

That is a legitimate probe, and it was chosen deliberately — a harmless-but-ambiguous animal is
exactly the stimulus a vigilance claim needs. But it is not the world the ladder measured, and it is
not a world either agent trained in.

## 3. Are they the same measure and window?

**No, on both counts — and the window alone is enough to flip the ladder's own answer.**

Both use the same behavioural quantity, "share of steps spent standing in a bush", so that much is
common. They differ in the stretch of episode and in the wound variable.

I recomputed the ladder's own dose-response at three windows and, additionally, split by whether the
episode contained a hunting predator at all. This used the current analysis code, including the
1 September correction that reads a predictor from the previous row. The pooled column reproduces the
published table exactly, which is the check that the recomputation is faithful.

**Change in bush hiding from the lightest to the heaviest quarter of the randomised starting wound,
in percentage points. Positive = hides more when wounded.**

| arm | first 25 steps | first 100 steps | whole episode |
|---|---|---|---|
| `A_baseline` | **−2.34** | −4.28 | −1.98 |
| `B_olf_only` | +3.38 | −0.84 | −0.76 |
| `R1_range1` | +4.15 | −0.44 | −0.49 |
| `V1_blur40` | +2.63 | −0.50 | −0.49 |
| `V2_blur20` | +2.57 | −0.60 | −0.55 |
| `V3_blur10` | +3.46 | −0.46 | −0.51 |
| `V4_blur05` | +4.00 | −0.23 | −0.41 |
| `V5_sharp` | +5.38 | +0.40 | −0.17 |
| `P1_blur05_iso` | +3.59 | −0.08 | −0.42 |
| `Q1_presence_sum` | +5.22 | −0.16 | −0.49 |
| `Q2_presence_binary` | +4.84 | −0.00 | −0.32 |
| `O1_occl_rock` | +6.69 | +0.59 | +0.00 |
| `O2_occl_veg` | +5.12 | −0.21 | −0.36 |
| `O3_occl_all` | +5.41 | −0.01 | −0.37 |
| **positive in** | **13 of 14** | **2 of 14** | **1 of 14 (by +0.003)** |

At the ladder's own 25-step window the answer is "hides more" in thirteen arms. Widen the window to
the length of the bush-refuge probe and it is "hides more" in two; take the whole episode and the
largest positive value across all fourteen arms is **+0.003** percentage points, which is zero.
**Measured over a comparable stretch of the episode, the sensor ladder gives the bush-refuge
answer.**

The third relevant study — the same factor analysis run on the bush-refuge agents *in their own
training world*, the closest apples-to-apples comparison available — reports the starting wound at
−1.02 percentage points per standard deviation for the arm it analyses, and −0.94 to −1.28 across
the other nine. Same sign as the ladder's whole-episode column, same order of magnitude. **In the
training world, over a whole episode, all twenty-four agents across both lines agree: a wound the
agent woke up with slightly reduces cover use.**

**Where the two studies already agreed and nobody noticed.** On the *felt* wound — the delayed,
smoothed internal signal the agent actually receives, as opposed to the wound in its body — both
lines report a strong positive relationship, on different runs in different worlds:

| study | bush hiding at the lowest felt-wound level | at the highest |
|---|---|---|
| sensor ladder (14 arms) | 12.6–17.9% | 31.7–49.5% |
| bush-refuge factor analysis (no predator within two cells) | 11.4% | 31.0% |

Roughly a threefold rise in both. This is the strongest point of agreement between the two lines and
it survives every difference catalogued above.

**One caution about the bush-refuge claim's own numbers.** I reproduced its headline: averaged over
the last quarter of each run's checkpoints, unwounded-minus-wounded bush use is +5.9 percentage
points, positive in 10 of 10 arms, in the harmless-rabbit scene. But the *same* comparison run in the
scene that contains a real predator gives +6.4 points — which looks like the effect generalises, and
does not, because the wounded agent dies far sooner there (66–87 steps against 97–101), so the two
sides are averaged over differently-selected steps. The original analysis caught this and matched on
time, at which point the predator-scene effect vanishes. And in the scene with no animal at all, the
gap is +0.01 points on a base of 0–4%: with nothing to hide from, there is no signal either way. **The
"hides less" claim is specific to a no-threat scene in which the agent nonetheless uses cover.**

## 4. One behaviour, two scoreboards

The three sections above establish that the claims do not meet. This section is the part that was
not in either study: what the agent is *actually doing*.

I recomputed, for all fourteen ladder arms over the first 25 steps, how often the agent chose the
**Rest** action, and split cover occupancy into resting-in-cover and in-cover-while-acting.

**Finding 1 — a wound makes the agent stop moving, in every arm, by a very large margin.**
Rest-action rate from the lightest to the heaviest starting-wound quarter rises by **+17.7 to +35.2
percentage points**, in 14 of 14 arms, in both the predator-present and predator-absent conditions.
In the reference arm with no predator present it goes from 22.8% of steps to 55.8%. The ladder study
never reported this; it is the largest wound-driven behavioural change in the data by roughly a
factor of five.

This is exactly the mechanism the bush-refuge line identified and named: injury heals **only**
through the Rest action, and healing is directly rewarded, so a wounded agent's best move is to
freeze and heal.

**Finding 2 — the ladder's "extra hiding" is entirely resting-in-cover, and active use of cover
falls.** Decomposing the change in bush hiding (percentage points, lightest to heaviest wound
quarter, first 25 steps):

| condition | change in bush hiding | of which: resting in a bush | resting **out** in the open | in a bush while **acting** |
|---|---|---|---|---|
| predator present, 14 arms | −2.96 … +9.50 | +0.59 … +14.76 | +10.6 … +20.2 | **−1.24 … −9.49, negative in 14/14** |
| no predator, 14 arms | +0.82 … +7.31 | +1.89 … +8.75 | +18.8 … +31.9 | **−0.77 … −1.79, negative in 14/14** |

Read the last column. **In every arm, in every condition, the agent occupies cover *less* when
wounded whenever it is actually doing something.** All of the ladder's positive dose-response — and
more — is accounted for by the agent sitting still, on a square that happens to be a bush.

Note the middle two columns' relative sizes: the extra resting lands overwhelmingly **out in the
open** (+18.8 to +31.9 points with no predator) rather than in a bush (+1.9 to +8.8). The wounded
ladder agent is not travelling to cover. It is freezing where it stands, and there are four to ten
bushes scattered over a hundred squares, so a modest slice of that freezing lands on one.

**The reconciliation.** Put the same behaviour into the bush-refuge probe. The agent starts three
squares from the *only* bush in the scene. Freezing there means never reaching cover, so bush hiding
collapses — the original analysis measured exactly that: while wounded, the agent chooses Rest on 86%
of steps and is in the bush on 1–3% of them. Meanwhile the unwounded comparison agent, with no food
to forage for and nothing else to do, walks over and sits in the bush (11–28%). The gap is large and
negative.

So:

- In a world with cover scattered everywhere and food to compete for it, **freezing raises measured
  cover use**.
- In a scene with one patch of cover three steps away and no competing activity, **freezing lowers
  measured cover use**.

Same policy. Same wound. Same primary response. Opposite sign on the scoreboard, because the
scoreboard is measuring position and the behaviour is about motion.

## 5. What each claim is actually a claim about

Phrased so both can appear in the same document without contradiction:

> **The sensor-ladder claim** is that in its own training world, over the twenty-five steps in which
> a randomly assigned wound is still being felt, a wounded agent spends a larger share of its steps
> standing in a bush than an unwounded one — a by-product of its choosing to stop and rest amid
> scattered cover, not evidence that it travels to cover when hurt.

> **The bush-refuge claim** is that in a purpose-built test scene containing one harmless animal, no
> predator, no food and a single patch of cover three squares away, a wounded agent reaches that
> cover far less often than an unwounded one across the whole hundred-step episode — because healing
> requires holding still and the agent holds still where it stands.

And the sentence both support:

> **A wound makes this agent stop moving and rest. Whether that registers as more or less hiding is
> a property of where cover is, not of a decision to hide.**

## 6. What this changes about how the sensor-ladder study should be read

Three adjustments, in descending order of importance. None of the ladder's numbers are wrong; the
reading attached to one of them is.

1. **Finding 5's "extra hiding" should be described as extra resting that lands in cover, not as a
   defensive response.** The ladder's Section 5 already stops short of calling it hypervigilance and
   calls it "ordinary caution". The decomposition above says it is not caution either: cover use
   while *acting* falls with the wound in 14 of 14 arms. The word "hiding" carries an intention the
   data does not support here.
2. **Limitation 2 should be broadened.** It currently records that the *hypervigilance* measure
   changes sign at a 50-step window. The **dose-response itself** changes sign too: positive in 13 of
   14 arms at 25 steps, and no larger than +0.003 percentage points in any arm over the whole
   episode. Figure 14 panel B shows this, but
   the limitation does not name it, and the limitation is what a hurried reader reads.
3. **The apparent conflict with the bush-refuge line should be named and dismissed in the study
   itself**, so the next reader does not have to redo this. A one-line pointer to this document is
   enough.

The ladder's headline sensory findings — directional smell is worth more than sharper sight, identity
matters more than sharpness, the agents that hide most die soonest — are untouched by any of this.

## 7. Limitations of this analysis

1. **Post-hoc.** Nothing here was pre-registered. I chose which comparisons to run after reading both
   claims, and the rest-decomposition in §4 was designed after seeing the window table.
2. **Single seed everywhere.** All twenty-four agents across both lines trained from seed 42. The
   14-of-14 and 10-of-10 unanimities are unanimity across *configurations*, not across independent
   training draws, and the ladder's own Limitation 1 applies in full.
3. **§4's decomposition is descriptive, not causal for the split.** The starting wound is randomised,
   so its total effect on resting and on bush hiding is causal. Splitting that effect into
   "resting-in-cover" and "in-cover-while-acting" conditions on the agent's own action, which is not
   randomised. The unanimity of the sign across 14 arms and both threat conditions is what the claim
   rests on, not the size of any one number.
4. **I did not run the corresponding decomposition on the bush-refuge probe recordings.** The
   original analysis reports the ingredients (86% rest, 1–3% in bush while wounded) and I took them
   as given rather than recomputing from its recordings. A step-level pass over those recordings
   would close the loop.
5. **The felt-wound agreement in §3 is associational on both sides.** An agent feels hurt because it
   was hurt, which depends on what it was doing. Both studies label it as such.
6. **No training was run, and nothing under `src/` or `configs/` was modified.**

## 8. Methods

**What was read.** Each run's own saved settings file (`models/config.yaml`) rather than a fresh
resolution of the source configs, so that what is compared is what the trainer actually used. The
ladder's collected trajectory stores (1,000,000 episodes per arm, two contiguous collection passes).
The bush-refuge probe's aggregated per-checkpoint result tables.

**What was recomputed.** A single sweep over all fourteen ladder arms' step-level stores, at
`tmp/ladder_threat_split.py`, accumulating bush occupancy, Rest-action rate and their intersection
into a grid of (starting-wound quarter × predator present or absent × outcome window). It copies
three conventions verbatim from the study's own `scripts/analysis/ladder/build_arm_data.py`: the
`t=0` row is not a step and appears in no numerator or denominator; the starting wound is read from
that `t=0` row; and "first 25 steps" means steps 1 to 25. It asserts, as the original does, that the
step counts it accumulates match the episode table's own length column, and that the two collection
passes form one contiguous block of seeds. **Its pooled column reproduces the study's published
Figure 8 and Figure 14 numbers to the last printed digit**, which is the check that it is faithful.

Run on the current code as of commit `92ba12c2`, which includes the 1 September fix (`c20ff5db`)
that makes the felt-wound sweep read its predictor from the previous row.

**Independent reproduction of the bush-refuge headline.** Averaged over the last 25% of each arm's
checkpoints, paired per checkpoint, from the probe's own result tables: unwounded-minus-wounded bush
use +5.91 points on average, positive in 10 of 10 arms, in the wandering-rabbit scene. Matches the
recorded value.

**Working files** (not committed): `tmp/20260903_1445_injury_hiding_sign.md`,
`tmp/ladder_threat_split.py`, `tmp/ladder_threat_split/*.json`.

## 9. Metrics Requested

| | |
|---|---|
| **Metric** | `rest_action_rate` and `rest_in_cover_rate` — share of steps on which the agent chose Rest, and the share on which it chose Rest while standing in a bush. Both unitless fractions. |
| **Why now** | The single largest wound-driven behavioural change in this data (+18 to +35 percentage points) is invisible in every summary this project currently produces, which is why two studies could publish opposite signs on a derived quantity without either noticing they were both watching the same freeze response. Bush hiding alone cannot distinguish "went to cover" from "stopped moving on a square that was cover". |
| **Where it'd live** | Alongside the existing bush-hiding measure in the behaviour-measure toolkit, and as a standard column in the dwell-sweep probe aggregation. The raw ingredients (`rested`, `agent_in_bush`) are already recorded per step in the trajectory stores, so this is a reporting gap rather than a logging gap. |
| **Cost** | Cheap. Two scalars per step, both already computed by the environment. |

| | |
|---|---|
| **Metric** | `dist_to_nearest_active_bush` at each step (Chebyshev). |
| **Why now** | §4's reconciliation turns on how far cover is when the agent freezes — three fixed squares in the probe, versus a random draw over 4–10 scattered bushes in the training world. That distance is currently reconstructable from raw position columns but is not a first-class measure, so the quantity that decides the sign of the headline is the one nobody plots. |
| **Where it'd live** | The same behaviour-measure toolkit; derived from the obstacle position columns already in the stores. |
| **Cost** | Cheap for the online logger; moderate as a store column (one small integer per step). |

## 10. Related Issues

Neither is a bug in code; both are recorded observations for the user to route.

1. **The rest-premium probe evaluates every arm under the default healing regime, not its own.** The
   probe scenes set only the starting wound and hunger and inherit everything else from the project
   default, so all ten arms — trained under continuity bonuses spanning a factor of ~130,000 — were
   measured under the default bonus. The sweep's own design matched the *length* of the wounded
   window across arms, so this is a mild mismatch rather than a broken comparison, and probing every
   arm under one common regime is a defensible way to isolate policy differences from dynamics
   differences. But it does mean the probe cannot express the variable the sweep manipulated, which
   is worth stating next to that sweep's null result. Worth a line in the bug/known-issues registry
   as a design note.
2. **The sensor-ladder study needs the three edits in §6.** These are documentation changes to a
   published study, not code, and belong to whoever owns that document.

## 11. Links

- The sensor-ladder study: [`sensor_ladder`](../sensor_ladder/sensor_ladder.md) — the source of the
  "hides more" claim (its Figures 8, 9 and 14).
- The in-training-world factor analysis of the bush-refuge agents:
  [`a01_hiding_drivers`](../trajectory_factors/a01_hiding_drivers.md) — the closest apples-to-apples
  comparison, and the third data point in §3.
- The origin of the "hides less" claim, in the project's insight wiki:
  [`20260810_1753_bushrefuge_injury_suppresses_bush_use_heal_by_rest`](../../../llm_wiki/entries/behavior_measures/20260810_1753_bushrefuge_injury_suppresses_bush_use_heal_by_rest.md)
  (the finding and its heal-by-rest mechanism),
  [`20260818_1620_rest_premium_sweep_refuted`](../../../llm_wiki/entries/behavior_measures/20260818_1620_rest_premium_sweep_refuted.md)
  (failed intervention 1 — re-pricing continuous rest), and
  [`20260819_1945_ambush_risk_refuted_normalize_gap_floor_effect`](../../../llm_wiki/entries/behavior_measures/20260819_1945_ambush_risk_refuted_normalize_gap_floor_effect.md)
  (failed intervention 2 — removing ambush hazards).
- The daily log entry that put the two claims side by side:
  [`2026-09-01`](../../../diary/2026-09-01.md), session `96e71c7b`, "What's next" item 4.
- The Principal Investigator's note asking for this check:
  [`2026-09-01_sensor_ladder_replication`](../../../pi/calls/2026-09-01_sensor_ladder_replication.md), §6.
