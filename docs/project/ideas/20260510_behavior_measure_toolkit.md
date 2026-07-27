---
title: "Fine-grained behavior-measure toolkit (event-triggered, time-locked, motif-level)"
topic: behavior
status: draft
created: 2026-05-10
last_updated: 2026-05-10
---

# Fine-grained behavior-measure toolkit — first-pass idea memo

## 1. Purpose

For most of this project, behavior has been read off **mean-statistics**: average distance to predator vs. rabbit, hit counts per episode, survival steps. The just-resolved sameProp study showed why that is not enough. The agent's apparent class avoidance was actually **location-conditional corner-camping** — under matched smells the agent never visits the dangerous corner at all, so the predator looks far on average even though the same-corner predator and the same-corner rabbit are treated identically. The per-tag distance metric the project shipped this week recovered the local picture and flipped the verdict, but it is still a *spatial*, *episode-mean* statistic. It cannot tell us whether the agent **interrupts feeding** when a predator approaches, whether it **freezes**, **flees**, or **dives into a bush**, or whether the response **habituates** when a rabbit comes back a second time.

The user's anchor example names the gap precisely: when the agent is eating and a predator (vs. a rabbit) enters its olfactory radius, does it stop eating and dive into a bush, just shuffle off the food, or carry on? That is a class-conditional defensive-behavior measure, and a paper that wants to claim "the modulator changes the agent's relationship to threat" has to show it. This memo lays out a paper-ready toolkit of **6–10 candidate measures** that turn trajectories into figure-grade behavioural readouts. The measures are organised by category (event-triggered counters, time-locked windows, conditional distributions, trajectory motifs, risk-discounted foraging, habituation traces) and each is tagged for online (cheap WandB scalar) vs. offline (eval-rollout dump + post-hoc analysis) placement. The memo ends with a curated 3-4-measure subset that the `experiment-designer` can lift directly into the next sameProp / hypervigilance round and the upcoming NMN-vs-baseline comparison.

## 2. Operating constraints already settled

These were resolved with the user via `AskUserQuestion` upstream of this memo and define the design envelope:

- **Audience**: paper-ready toolkit. Lives in the codebase long-term. Used across hypervigilance, NMN, and future studies.
- **Data source**: BOTH paths land. **Online** = lightweight per-episode scalar counters that WandB can plot during training (one `Episode/<measure>` key per measure, additive to the existing per-tag fan-out). **Offline** = an eval-rollout script loads a frozen checkpoint, runs N test episodes against a held-out seed set, dumps full `(s, a, info)` trajectories to disk for post-hoc motif analysis and time-locked plotting.
- **Geometry-agnostic**: the measures consume per-tag and per-instance keys (the existing `tag` field on each entity), never quadrant geometry. A measure written today on a 4-quadrant 10×10 grid must work unchanged when Round 3 spawns food in all four quadrants or when a future study uses a 9-cell arena.
- **No fallback defaults**: any new YAML key follows `config.get_mandatory(...)` per project rules. Measure-specific knobs (radius `R`, window `K`, etc.) are explicit in the experiment config.

## 3. What the env already exposes per step

A read of `src/environment/core.py` confirms every measure below is computable from quantities the env already emits per step (`info` dict around line 432–515) plus the agent state. The relevant pre-existing handles are:

- `info['ate_food']` (bool) — fired the step the agent ate food (auto or via `eat` action).
- `info['hit_predator']`, `info['hit_neutral']`, `info['hit_hiding_predator']` (bools) — contact events.
- `info['damage_*']` family — per-source damage, useful for distinguishing predator-damage from rock-damage in motifs.
- `info['dist_to_food']`, `info['dist_to_pred']`, `info['dist_to_neutral']`, `info['dist_to_hiding_predator']` (scalars, min over instances).
- `info['dist_per_neutral']`, `info['dist_per_predator']` (`[num_*]` vectors) — the per-tag work shipped this week. **Every measure below conditions on these per-instance distances, not on quadrant geometry.**
- `info['event_collided']` (bool) — wall / blocking-rock collision.
- `info['drive_hunger']`, `info['drive_injury']` — homeostatic state, useful as conditioners ("eating events when hunger is high vs. low").
- `state.agent_pos`, `state.last_action`, and the `params.obs_hides_agent` mask — together identify *bush-occupancy* and *eat-action* events at zero env-side cost.

**One env-side hook is needed** for the motif-level measures: surface a per-step `info['agent_in_bush']` boolean (one line: indexes `params.obs_hides_agent` at `state.agent_pos`). This is mechanically trivial; named in §6 as a `senior-developer` ticket. **No other env hooks are needed** — the rest is `train.py` accumulator work plus an offline eval-rollout script.

## 4. The measure menu

Eight candidates, organised by category. Each entry: *what it captures*, *operationalisation*, *online vs offline*, *grounding*, *what it would buy us in the recent / upcoming experiments*.

### 4.1 Event-triggered counters (online)

#### M1 — `interrupted_feeding_rate_<class>`

**What it captures.** The user's anchor example formalised. When the agent is on a food cell, a class-`c` entity within radius `R` should change the agent's next action from "stay & eat" to "leave & seek cover". The rate at which it does is a class-conditional defensive-behavior signal.

**Operationalisation.** For each step `t` where `info['ate_food'][t]` is True (the agent is currently eating) AND `min(dist_per_<class>[t]) < R` (a class-`c` entity is within the cue radius), record a candidate event. Within the next `K` steps, fire the predicate `interrupted := (info['ate_food'][t+1:t+K+1] == 0).any()` (the agent stops eating in the K-window). The measure is `Σ interrupted / Σ candidates` aggregated per episode. Two scalars per episode, one per class: `interrupted_feeding_rate_predator`, `interrupted_feeding_rate_rabbit`. Use per-tag splits when more than one tag of a class exists: `interrupted_feeding_rate_predator_<tag>`. `R` and `K` are mandatory YAML knobs; suggested defaults `R = 3.0` cells, `K = 5` steps.

**Placement.** Online. Three integer accumulators per class per episode (`candidates`, `interrupted`, `aggregated_distance_at_interrupt`), zero JAX changes — `info['ate_food']` and `dist_per_*` already vectorise. Survives the existing 5-site fan-out unchanged.

**Grounding.** The behavioral-ecology literature on "feeding interruption under predation risk" (Lima & Dill 1990 family) is the canonical reference. The animal model is: a foraging vole that drops a peanut and dives for cover when a hawk shadow passes. Drosophila and zebrafish neuroethology use the same primitive ("startle-from-feeding rate"). For RL agents, this is a direct analogue of "policy switches sub-goal under threat" — the measure is the empirical proxy for what the active-inference / hypervigilance frame would predict.

**What it buys us.** *Directly relevant to Round 3.* In sameProp Round 2.5 the mean-distance metric was dominated by corner-camping; an interrupted-feeding-rate metric would have caught a specifically class-conditional behavior even on the camping basin (the agent does still eat occasionally — the question is *what makes it stop*). Generalises to NMN-vs-baseline as the headline behavioural figure: "modulated agents interrupt feeding 2.3× more often when the cue is a predator than when the cue is a rabbit; baseline agents show no class asymmetry."

#### M2 — `bush_dive_rate_<class>`

**What it captures.** Of all "threat enters radius `R`" events, what fraction triggers a *bush dive* (agent's next-K trajectory ends on a `hides_agent` cell)? This is the "dive into bush in case of predator" half of the user's example.

**Operationalisation.** Mark each step `t` where `min(dist_per_<class>[t]) < R` as a candidate (bool predicate, per class). Within `t+1 : t+K`, fire if `info['agent_in_bush'][t+1:t+K+1].any()` AND the agent was not already in a bush at step `t`. Episode scalar: `bush_dive_rate_<class> = Σ fired / Σ candidates`. Per-tag splits available. Suggested defaults match M1 (`R = 3.0`, `K = 5`).

**Placement.** Online. **Requires the new `info['agent_in_bush']` env hook** (one line in `core.py`: `info['agent_in_bush'] = params.obs_hides_agent[state.agent_pos[0], state.agent_pos[1]]` or equivalent; the obstacle index lookup pattern is already used elsewhere in the file).

**Grounding.** "Refuge use" / "cover-seeking under threat" is the standard ethological readout for actively defensive behavior — distinct from freezing (passive immobility) and flight (active distance-increasing locomotion). Fanselow's defensive-behavior taxonomy maps cleanly: cover-seeking is the high-cost / high-safety response; freezing is low-cost / medium-safety; flight is high-cost / cue-dependent.

**What it buys us.** Discriminates *active defense* from *spatial avoidance*. Corner-camping yields zero bush-dives because the agent never sees a threat enter its radius. NMN agents that genuinely respond to the predator class (rather than just camping a different corner) should have non-zero `bush_dive_rate_predator` and approximately zero `bush_dive_rate_rabbit`. This is the cleanest single number for "the agent does class-conditional active defense", and it is the one most worth a panel in the paper.

#### M3 — `flight_initiation_distance_<class>`

**What it captures.** Animal-behavior's "flight initiation distance" (FID): the distance at which an animal abandons its current activity and begins moving away from an approaching threat. Single-number summary of how *risk-averse* the agent is per class.

**Operationalisation.** Slide a window forward through each episode. For each step `t` where `min(dist_per_<class>[t])` is decreasing (threat approaching), mark the first step `t*` where `||agent_pos[t* + 1] - threat_pos[t* + 1]|| > ||agent_pos[t*] - threat_pos[t*]||` (distance increases — agent retreats). Record `min(dist_per_<class>[t*])` as the FID for that approach episode. Episode-mean per class: `flight_initiation_distance_<class>`. Suggested implementation: only count approaches where the threat closed by ≥ 2 cells before the flight (rules out one-step random jitters).

**Placement.** Online (single scalar per class per episode). Slightly more involved than M1/M2 — needs a per-env "currently-tracking-an-approach" state machine — but doable in NumPy in `train.py` over the existing per-step accumulator. If the bookkeeping is too brittle online, demote to offline; both versions land cleanly.

**Grounding.** FID is the most-cited single number in field ecology of antipredator behavior (Ydenberg & Dill 1986; Stankowich & Blumstein meta-analysis 2005). It is directly proportional to perceived risk and inversely proportional to the value of the activity being interrupted.

**What it buys us.** Continuous, comparable across studies. A figure that plots `FID_predator − FID_rabbit` as the y-axis across training time tells a clean "the modulated agent learns to flee predators sooner than rabbits" story. In Round 2.5 Cell A1 this would have been ~0 (the agent never lets either entity approach because it camps), correctly diagnosing the corner-camping confound without per-tag distance.

### 4.2 Time-locked windows (online for small K, offline for state-rich windows)

#### M4 — `peri_threat_action_distribution_<class>`

**What it captures.** When a class-`c` threat first crosses radius `R`, what does the agent's action distribution look like over the next `K` steps? Compared against a baseline action distribution computed when no threat is within radius. The difference is a *class-conditional behavioral signature*.

**Operationalisation.** Mark each step `t*` where `min(dist_per_<class>[t* - 1]) ≥ R` AND `min(dist_per_<class>[t*]) < R` as a "threat onset". Collect the per-step actions over `[t*, t* + K]` into a 6-bin histogram (the env's 6-action space). Online version: log only the **mode action** and `Σ stays_in_place / K` (one scalar per onset, mean across onsets per episode). Offline version: dump the full action histogram per onset for figure-grade plotting — useful for "we see a flight burst at lag 1, then a freeze at lag 4" stories.

**Placement.** Both. Online scalar version goes straight into WandB; offline trajectory dump enables the figure in §5.

**Grounding.** "Peri-event time histogram" is the canonical neuroscience tool for time-locked analysis (event-related potentials, peri-stimulus spike rasters). Direct transfer to behavior: peri-threat-onset action distributions are the behavioral PETHs of an RL agent. Fear-conditioning literature (LeDoux, Maren) uses exactly this windowing to distinguish freezing from escape.

**What it buys us.** Distinguishes between policies that look identical on mean-distance but differ on *immediate response shape*. Two agents both keeping predators 4 cells away on average could be doing it via "constant slow drift away" (no peri-event response) vs. "rapid 2-step retreat then rest" (sharp peri-event response). The latter is the only one that supports a "modulator triggers a defensive routine" claim.

### 4.3 Conditional action / state distributions (online ratios, offline for state-rich)

#### M5 — `eat_under_threat_ratio`

**What it captures.** Optimal foraging under predation: `P(eat | min(dist_per_predator) < R) / P(eat | min(dist_per_predator) ≥ R)`. A risk-discounting agent should eat *less* under threat (ratio < 1); a risk-blind agent eats at the same rate (ratio ≈ 1); an erratic / panicked agent might even eat more under threat (ratio > 1, pathological).

**Operationalisation.** Two episode-summed counters: `eats_under_threat`, `steps_under_threat`, `eats_safe`, `steps_safe`. Episode scalar: `(eats_under_threat / max(steps_under_threat, 1)) / (eats_safe / max(steps_safe, 1))`. Class-split versions: `eat_under_threat_ratio_predator`, `eat_under_threat_ratio_rabbit`. Both numerator and denominator emitted to WandB for sanity.

**Placement.** Online. Four NumPy counters; no JAX changes; lives in the existing per-step accumulator block.

**Grounding.** Optimal foraging theory under predation (Lima 1998; Brown 1999) — the giving-up density and risk-foraging trade-off. Directly maps to the project's pain-modeling framing: a "hypervigilant" agent over-discounts food when threat is non-zero.

**What it buys us.** Single number that captures a paper-thesis-relevant trade-off. NMN vs. baseline figure: "modulated agents have `eat_under_threat_ratio_predator = 0.4` and `eat_under_threat_ratio_rabbit = 1.0`; baseline has both ≈ 0.9". Directly testable, directly publishable.

### 4.4 Habituation / sensitisation (online as time-series, offline for figure)

#### M6 — `peri_threat_response_decay_<class>`

**What it captures.** Within a single episode, the agent encounters the same threat instance multiple times. Does the defensive response (FID, bush-dive, eat-suppression) **weaken** with repetition (habituation, the healthy pattern)? **Strengthen** (sensitisation, the hypervigilant pattern)? **Stay flat** (no learning within episode)?

**Operationalisation.** For each episode, partition threat-onset events (M4 definition) by their ordinal position in the episode (1st, 2nd, 3rd encounter, …). Compute the M2 `bush_dive_rate` or M5 eat-suppression on each ordinal subset. The episode scalar is the **slope** of the response across ordinal positions: `slope = (response[3+] − response[1])`. Negative slope = habituation; positive = sensitisation; flat = no within-episode learning. Per-class.

**Placement.** Both. Online version logs only the slope (single scalar per episode); offline version logs the full ordinal-by-ordinal response curve for the figure.

**Grounding.** Habituation vs. sensitisation is the most basic learning dichotomy in behavioral neuroscience (Aplysia gill-withdrawal — Kandel; mammalian startle-response — Davis). Sensitisation under threat is the canonical animal-model signature of "stress-induced hypervigilance" — directly relevant to the project's pain-modeling thesis.

**What it buys us.** A directional measure. If NMN agents show negative slope (habituation) on rabbits and ≈ 0 slope (no habituation) on predators, that is *exactly* the pattern a healthy hypervigilance circuit predicts. If they show positive slope on rabbits (sensitisation to a non-threat), that is the pathological-hypervigilance pattern the project is trying to model. **This is a paper-thesis-grade measure** — it doesn't just measure behavior, it diagnoses *which kind of dysregulation* the agent is exhibiting.

### 4.5 Trajectory motifs (offline only)

#### M7 — `defensive_motif_repertoire`

**What it captures.** Cluster the agent's `K`-step `(state-summary, action)` trajectories around threat-onset events into a small number of canonical motifs: `freeze`, `flight`, `bush_dive`, `freeze_then_flight`, `ignore`, `approach` (the worrying one). The episode-level / checkpoint-level readout is the **motif distribution** — what fraction of the agent's threat-encounters fall into each cluster.

**Operationalisation.** Offline only. (1) Run the eval-rollout script: load a frozen checkpoint, roll out `N = 200` episodes against a held-out seed set, dump `(agent_pos[t-2:t+K], action[t:t+K], dist_per_*[t-2:t+K], info['ate_food'][t:t+K], info['agent_in_bush'][t:t+K])` for every threat-onset event into a parquet file. (2) Featurise each event as a vector `(net_displacement, max_dist_change, action_entropy, mean_eat_rate, fraction_in_bush, ...)`. (3) Cluster (k-means with k = 5–7 OR HDBSCAN) the events into motifs. (4) Label each cluster manually once (the figure caption); thereafter the motif assignments are stable.

**Placement.** Offline. Requires the eval-rollout script (named in §6 as a `senior-developer` ticket). Output is one parquet per checkpoint per condition; one figure per study.

**Grounding.** Behavioral-motif clustering on trajectory data is now standard in computational ethology (MotionMapper / Berman 2014; B-SOiD; SimBA). The defensive-behavior taxonomy (freeze / flight / refuge / ignore) maps onto the motif vocabulary across vertebrates from fish to mice.

**What it buys us.** The figure-grade story for a paper. "Baseline agents threat-encounters cluster mostly into `ignore` (47%) and `flight` (38%); NMN agents cluster mostly into `bush_dive` (52%) and `flight_then_freeze` (29%)." That is the kind of behavioral phenotype a reviewer can hold and trust. It also gives a *qualitative* readout, which the mean-statistics never can.

### 4.6 Risk-discounted foraging (online sum, offline for decomposition)

#### M8 — `food_intake_under_risk_curve`

**What it captures.** The cumulative-food-intake-vs-cumulative-time-under-risk curve, decomposed across episodes. Steeper slope = more food eaten per unit of time spent in radius-`R` predator proximity = more risk-tolerant (or more risk-blind) policy.

**Operationalisation.** Two cumulative episode counters: `food_eaten` (sum of `info['ate_food']`) and `steps_under_predator_threat` (sum of `min(dist_per_predator) < R`). Episode scalar: `food_eaten / max(steps_under_predator_threat, 1)`. Plot across training as a learning curve. Offline version: dump the full per-episode `(food_eaten, steps_under_threat, steps_safe)` triple for figure-grade decomposition (e.g., scatter coloured by checkpoint epoch).

**Placement.** Both. Online ratio is one scalar per episode; offline tuple supports the decomposition figure.

**Grounding.** Marginal-value-theorem style trade-off measurement. Direct analogue of "patch residence time as a function of patch quality and predation risk" (Charnov 1976; Brown 1988). The decomposition is the project's pain-modeling thesis in behavioral form: "this agent is over-discounting food intake under non-zero risk, even when the risk has a low base rate".

**What it buys us.** Generalises across studies (sameProp, NMN, future). Cross-study figure: `food_intake_under_risk_curve` for plain RPPO vs. NMN-FiLM vs. NMN-FiLM-modulator-clamped — three lines on one plot. If they separate, that is the project's headline behavioural finding.

## 5. Paper-ready figure mapping

| Measure | Hypervigilance figure | NMN figure | Cross-study figure | Single-figure standalone? |
|---|---|---|---|---|
| **M1 — interrupted_feeding_rate** | Yes (Round 3, class-conditional) | Yes (NMN vs. baseline panel) | Yes (cross-study learning-curve) | Yes — single bar chart per class |
| **M2 — bush_dive_rate** | Yes (Round 3) | **Headline panel** | Yes | Yes — bar chart |
| **M3 — flight_initiation_distance** | Maybe (low priority for camping configs) | Yes | Yes (FID-vs-training-step learning curve) | Yes — FID-vs-class scatter |
| **M4 — peri_threat_action_distribution** | Useful for diagnosing Cell A1 transient | Yes (PETH per class) | Maybe | Yes — peri-event PETH plot |
| **M5 — eat_under_threat_ratio** | Yes | Yes | **Cross-study headline scalar** | Yes — single bar per class per condition |
| **M6 — peri_threat_response_decay** | Yes (Round 3 only) | **Pain-modeling headline figure** | Yes | Yes — slope-by-class plot |
| **M7 — defensive_motif_repertoire** | Yes (qualitative phenotype panel) | **Phenotype figure** | Yes | Yes — stacked bar |
| **M8 — food_intake_under_risk_curve** | Yes | Yes | **Cross-study headline figure** | Yes — multi-line plot |

Read by columns:
- The hypervigilance Round 3 paper draws from M1 / M2 / M5 / M6 / M7.
- The NMN-vs-baseline paper draws from M2 (headline), M5, M6, M7, M8.
- The cross-study comparison (project-spanning) draws from M5, M8 as headline scalars and M1 / M2 / M3 as supporting curves.

## 6. Curated near-term subset (for `experiment-designer` handoff)

Three measures pay rent immediately on the next experiment cycle (Round 2.6 / Round 3 / NMN-vs-baseline). They are cheap to implement, paper-grade in their own right, and answer the immediate diagnostic question that mean-distance cannot: *does the agent do anything class-conditionally defensive at all?*

### N1 — M1 `interrupted_feeding_rate_<class>`  *(experiment-designer + senior-developer)*

Cheapest. Zero new env hooks, four NumPy counters, lands in the same five `train.py` accumulator sites that the per-tag work already touches. Highest leverage per line of code. **`experiment-designer` to author the spec (radius `R`, window `K`, YAML schema for the per-experiment knobs); `senior-developer` to plan the `train.py` implementation against the existing per-tag pattern.**

### N2 — M2 `bush_dive_rate_<class>`  *(senior-developer first, then experiment-designer)*

Requires the single-line `info['agent_in_bush']` hook in `core.py`. Once that lands, the rest is the same accumulator pattern as M1. **Most directly answers "does the agent do active class-conditional defense" — the headline question for the NMN paper.** Order: `senior-developer` plans the env hook + the `train.py` accumulator (single sub-task plan); after implementation lands, `experiment-designer` writes the experiment design that reads the new key.

### N3 — M5 `eat_under_threat_ratio_<class>`  *(experiment-designer + senior-developer)*

Symmetric to M1 in cost. Captures the optimal-foraging-under-predation trade-off at episode resolution, which generalises across studies and is a clean cross-paper headline scalar. Plan together with M1.

### N4 — M7 `defensive_motif_repertoire`  *(senior-developer plans the eval-rollout script)*

The qualitative phenotype figure — required for the NMN paper, optional for hypervigilance Round 3. **Implementation cost is the eval-rollout script (load checkpoint, run N episodes, dump trajectories), which is a one-time investment that pays back across every future experiment.** `senior-developer` to plan the script; `experiment-designer` to define the eval protocol (N episodes, seed set, checkpoint-selection rule) once the script lands.

**Implementation order recommendation**: N1 + N3 first (lowest cost, immediately useful in Round 2.6 / Round 3), N2 second (one env hook unlocks the headline NMN measure), N4 last (one-time script with broadest downstream payoff). Each as a separate implementation plan to keep commits surgical.

## 7. Open questions / tradeoffs (user to resolve before `experiment-designer` is spawned)

These choices are surfaced because they could reasonably go more than one way and picking silently risks rework:

1. **Cue-radius `R` and window `K`**: defaults proposed are `R = 3.0` cells and `K = 5` steps. `R = 3.0` matches the existing visual-sensor radius and the predator's `pred_detect = 5`-bounded signature; `K = 5` is half the agent's typical reaction window. Worth the user's call — wider `R` (e.g., `5.0`) catches more peri-threat events but mixes in noise; longer `K` (e.g., `10`) catches delayed responses but slips past the next threat onset. **Recommendation**: ship `R, K` as mandatory YAML knobs (per the no-fallback-defaults rule), default to `R = 3.0`, `K = 5` in the first sweep, and run a sensitivity sub-study if the headline measures are knife-edge.

2. **Trajectory dump rate vs. disk usage**: M7 / M4-offline require dumping full `(s, a, info)` per-episode. At 500 steps/episode × 200 eval episodes × 8 environments × ~50 floats per step ≈ 40 MB per checkpoint. Not free but well within the existing `wandb/` and `results/` budgets. **Recommendation**: dump *only* threat-onset windows (`[t-2, t+K]`), not full episodes — cuts the dump 30×. Decision is `senior-developer`'s in the eval-rollout script plan.

3. **Eval-protocol seed and N**: how many episodes per checkpoint, against which seed set? The existing per-tag training already uses 128 envs × ~1880 episodes per last-10% window — that's 240k episodes of "training-distribution behavior". The eval protocol should be **smaller and deterministic**: `N = 200` episodes against a fixed held-out seed set (seed list in the experiment config), so motif clusterings are reproducible across reruns. **User's call** whether to also offer a "rich-condition sweep" mode (loop over R, K, multiple checkpoints) for sensitivity analysis.

4. **Motif feature space**: M7 needs a featurisation. Options: (a) **handcrafted features** — net displacement, action entropy, fraction-in-bush, mean eat rate over the window — interpretable, robust, paper-grade. (b) **learned embedding** — feed the window through a small unsupervised encoder, cluster in latent space — more sensitive but reviewers will ask why this encoder. **Recommendation**: ship handcrafted first; add learned-embedding as a follow-up if the handcrafted clusters look noisy.

5. **Per-tag fan-out vs. per-class aggregation for online measures**: M1 / M2 / M3 / M5 can emit either per class (`_predator` / `_rabbit`) or per tag (`_predator_TL`, `_rabbit_TL`, `_rabbit_BR`). Per-tag is strictly more informative but adds 4–6 keys per measure per training run. **Recommendation**: emit per-class (sum over tags within a class) AND per-tag (each tag independently) — costs nothing extra in compute, and the per-tag version is what diagnoses "the agent treats the same-corner predator differently from the cross-corner rabbit" (the Round 2.5 lesson, applied to defensive measures rather than mean distances).

6. **Phase-2 per-cumulative measures vs. cohort-level summary**: should the toolkit emit per-episode scalars (the recommendation here, easy to plot) or running cumulative statistics over the full training window? **Recommendation**: per-episode is simplest and aligns with the existing `Episode/...` WandB key family. Cohort-level summaries can be computed offline from the per-episode dumps later.

## 8. Triage decision

**No professor triage.** The measures here are within standard ethology / RL / computational-neuroscience vocabulary; their grounding (Lima & Dill, Ydenberg & Dill, Fanselow's defensive taxonomy, Berman's motif clustering, Aplysia habituation, peri-event time histograms) is well-known and does not require a fresh concept memo from `professor-pain-modeling` to operationalise. Where pain-science grounding *will* be load-bearing is later — when we **interpret** the measured values, not when we **design** the measures.

The right time to invoke `professor-pain-modeling` is after the first NMN-vs-baseline run reads off these measures. The professor's job at that point is to map the measured pattern onto pain-science constructs ("the modulated agent's `peri_threat_response_decay` slope on rabbits is positive, matching central sensitisation in chronic-pain animal models") — which requires *real numbers* in hand. Inviting them now would generate a concept memo that re-derives the measures from a fear-conditioning / pain-modeling lens; useful but redundant against this memo's grounding paragraphs, and it would block the cheapest path to the next experiment.

The right time to invoke `professor-rl-bayesian-dl` is if M7 (motif clustering) needs a learned embedding rather than handcrafted features — that is genuinely an architecture / Bayesian-DL design call. Defer until the handcrafted clustering is run; if its quality is acceptable, no triage; if it isn't, route the embedding question to that professor.

## 9. References

- Sameprop Round 2.5 verdict insight (motivation): [`docs/llm_wiki/entries/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md`](../../../docs/llm_wiki/entries/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md).
- Study re-summary: [`docs/experiments/summaries/20260510_2253_sameprop_rabbit_avoidance_study.md`](../../experiments/summaries/20260510_2253_sameprop_rabbit_avoidance_study.md).
- Per-tag distance metric (the precedent the toolkit extends): [`docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md`](../../develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md).
- Discriminating-channels memo (what cues the agent could use): [`docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`](../../develop/active/hypervigilance/sameprop_discriminating_channels.md).
- Existing per-step `info` dict (env-side anchor): `src/environment/core.py` lines 432–515.
- Documentation framing rule (this memo's first body section): root `CLAUDE.md` §69–97.
- Frontmatter contract: [`docs/develop/active/meta/FRONTMATTER_CONTRACT.md`](../../develop/active/meta/FRONTMATTER_CONTRACT.md). (This memo lives under `docs/project/`, not `docs/develop/`, so the contract applies in spirit only — `topic: behavior` is a pre-existing valid topic.)
