---
title: "Behaviour Analysis of Deep RL Agents — Master Reference Review"
topic: behavior_analysis
status: in-progress
created: 2026-08-19
last_updated: 2026-08-19
papers_reviewed: 1
sources_held: 1
scope: |
  This folder collects work on **how to tell what a reinforcement-learning agent is
  actually doing**, using instruments richer than a reward curve. The target corpus
  is the intersection of (a) neuroscience / ethology behavioural-analysis methods —
  trajectory and path analysis, movement-state segmentation, choice regressions,
  occupancy statistics — and (b) neural-population analysis of an agent's internal
  state — linear decoding, per-unit encoding models, representational structure —
  applied jointly to artificial agents.

  Papers that merely propose a new RL algorithm and report learning curves are OUT
  of scope. Papers about interpretability of *supervised* networks are out of scope
  unless the method transfers to a temporally extended, partially observed agent.

  The folder exists because this repository's central methodological problem is the
  one this corpus names: performance in survival steps tells you *whether* an agent
  copes, and nothing about *how*.
related:
  - ../foraging_for_cognitive_evolution/
  - ../perceptual_decision_making/perceptual_decision_making_review.md
---

# Behaviour Analysis of Deep RL Agents — Master Reference Review

## 1. Plain-English entry point

When researchers train an artificial agent to survive in a simulated world, they
almost always judge it by one number: how much reward it collected, plotted against
how long it trained. That curve tells you whether the agent got better. It tells you
almost nothing about *what the agent learned to do* — whether it built a mental map,
whether it remembers where food was, whether it avoids a place because something bad
happened there, or whether it just stumbled into a lucky habit.

Biologists have this problem too, and have spent a century building instruments for
it: they film the animal, break its movement into recognisable behaviours, fit
statistical models to its choices, and record from its neurons while it behaves. The
paper reviewed here takes that toolkit off the shelf and points it at deep RL agents.

The authors built a foraging world — a large arena where food animals wander and run
out if you over-graze a spot, water sources are fixed, predators hunt you when they
see you, and you get hungry, thirsty and tired. They trained a standard, ordinary
agent in it: no world model, no planner, no explicit memory store, just a policy-
gradient learner with a small recurrent network for memory. Then they analysed it
like an animal.

What they found: the agent spontaneously explores in outward-spiralling loops that
look strikingly like a displaced desert ant searching for its nest, then switches to
efficiently returning to remembered food patches. Which patch it returns to is
predicted by how recently it visited, how much food it saw there, and whether a
predator attacked it there. And a simple linear readout of the agent's internal
memory can recover where the agent was up to a hundred steps ago and where it will be
up to a hundred steps ahead. The authors' headline claim is that planning-like
structure emerged from ordinary reward-driven training, with no planning machinery
built in.

The honest caveat — which the paper's own control experiment establishes and its
abstract does not mention — is that the spatial readout collapses to chance when the
authors remove a *supervised side-task* that explicitly trained the network to report
its own position. So the strongest version of the headline is not supported by the
paper's own data. That gap is the single most useful thing in this review for this
project, and Section 5 explains why this repository is well placed to close it.

---

## 2. How to read this document

Each paper gets four parts, in this order:

1. **Phase 1 — Foundational Overview.** Undergraduate level. The problem, the result,
   why it matters, no equations.
2. **Phase 2 — Graduate-Level Deep Dive.** Every formal object the paper defines,
   written out in full, with derivations rather than bare formulas, plus a critical
   assessment of what the evidence licenses.
3. **Relevance to this project.** What is portable into `grid_world_pain`, what data
   each method needs, what breaks, and which claims this repository can test.
4. **Appendix: Section-by-Section Backbone.** The paper walked through in its own
   section order, including appendices. This is the traceability layer — everything
   in Phases 1 and 2 is a reorganisation of material that appears here.

Verification convention: a claim tagged **[paper]** is stated by the authors; a claim
tagged **[reviewer]** is this review's own inference or arithmetic and has *not* been
checked against the authors. Do not quote a **[reviewer]** claim as the paper's.

## 3. Table of Contents

- [1. Plain-English entry point](#1-plain-english-entry-point)
- [2. How to read this document](#2-how-to-read-this-document)
- [3. Table of Contents](#3-table-of-contents)
- [4. Corpus manifest](#4-corpus-manifest)
- [5. Simmons-Edler et al. (2025) — Deep RL Needs Deep Behavior Analysis](#5-simmons-edler-et-al-2025--deep-rl-needs-deep-behavior-analysis)
  - [5.1 Metadata and provenance](#51-metadata-and-provenance)
  - [5.2 Phase 1 — Foundational Overview](#52-phase-1--foundational-overview)
  - [5.3 Phase 2 — Graduate-Level Deep Dive](#53-phase-2--graduate-level-deep-dive)
    - [5.3.1 The ForageWorld environment as a formal object](#531-the-forageworld-environment-as-a-formal-object)
    - [5.3.2 Observation and state](#532-observation-and-state)
    - [5.3.3 Agent architecture and objective](#533-agent-architecture-and-objective)
    - [5.3.4 Behavioural analysis methods](#534-behavioural-analysis-methods)
    - [5.3.5 Neural analysis methods](#535-neural-analysis-methods)
    - [5.3.6 Ablations, sweeps and the alternative algorithm](#536-ablations-sweeps-and-the-alternative-algorithm)
    - [5.3.7 What the evidence licenses — critical assessment](#537-what-the-evidence-licenses--critical-assessment)
  - [5.4 Relevance to this project](#54-relevance-to-this-project)
    - [5.4.1 Directly portable methods, and what data each needs](#541-directly-portable-methods-and-what-data-each-needs)
    - [5.4.2 Where the two environments differ enough to break a method](#542-where-the-two-environments-differ-enough-to-break-a-method)
    - [5.4.3 Claims this project is positioned to test or challenge](#543-claims-this-project-is-positioned-to-test-or-challenge)
  - [5.5 Appendix: Section-by-Section Backbone](#55-appendix-section-by-section-backbone)

## 4. Corpus manifest

| # | Short key | Citation | PDF |
|---|---|---|---|
| 1 | `simmons-edler-2025` | Simmons-Edler, Badman, Berg, Chua, Vastola, Lunger, Qian & Rajan (2025), *Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents in Open-Ended Environments*, NeurIPS 2025 | `docs/project/references/behavior_analysis/sources/Simmons-Edler et al. 2025 - Deep RL needs deep behavior analysis (arXiv v2 preprint).pdf` |

---

## 5. Simmons-Edler et al. (2025) — Deep RL Needs Deep Behavior Analysis

### 5.1 Metadata and provenance

- **Title:** *Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents in Open-Ended Environments*
- **Authors:** Riley Simmons-Edler\*, Ryan P. Badman\*, Felix Baastad Berg, Raymond Chua, John J. Vastola, Joshua Lunger, William Qian, Kanaka Rajan (\* equal contribution; Rajan corresponding)
- **Affiliations:** Harvard Medical School Department of Neurobiology; Kempner Institute, Harvard; NTNU; McGill & Mila; U. Toronto; Harvard Biophysics
- **Venue:** **NeurIPS 2025** (39th Conference on Neural Information Processing Systems)
- **Code:** <https://github.com/RileySE/Craftax-Foraging/tree/foraging>
- **arXiv:** [2506.06981](https://arxiv.org/abs/2506.06981)

> **Provenance caveat — read before citing verbatim.** The PDF held in
> `sources/` is the **arXiv v2 preprint** (`arXiv:2506.06981v2 [cs.AI]`, updated
> 2025-11-30), not the camera-ready of record. The camera-ready lives at
> <https://openreview.net/forum?id=QD06Qv7O0P> and was **not retrievable** at review
> time (Cloudflare challenge). Section numbering, figure numbering, table contents
> and equation numbering in this review therefore follow **v2**. The v2 posting
> post-dates the conference and carries the NeurIPS 2025 footer plus a completed
> NeurIPS paper checklist, so it is very likely at or beyond camera-ready content —
> but that has not been verified. Before lifting an equation number or a figure
> number into a submission, re-check against the OpenReview PDF.

---

### 5.2 Phase 1 — Foundational Overview

**The problem.** Deep reinforcement learning evaluates agents almost exclusively by
reward curves and aggregate scores. As the tasks get more open-ended — partially
observable, long-horizon, requiring memory and spatial reasoning — those curves stop
answering the questions people actually have: *what strategy did the agent find? what
does it remember? does it plan?* The authors argue this is a bottleneck, not a
cosmetic gap, because you cannot diagnose why an algorithm fails if you cannot
describe what it does. Neuroscience and ethology solved a structurally identical
problem for animals decades ago and built mature instruments for it; deep RL has
imported the *neural* half of that toolkit (representational similarity analysis and
friends) and largely ignored the *behavioural* half.

**The proposal.** Study RL agents the way ethologists study animals — behaviour
first, neural activity second, and always jointly. The paper contributes four things:

1. **ForageWorld**, a foraging environment built on top of Craftax (a fast,
   GPU-accelerated Minecraft-like 2-D benchmark). Its ecological additions are the
   point: food animals ("cows") are capped in number, spawn at fixed points and
   diffuse, so eating them out of a region creates a *temporary local famine* that
   forces the agent to leave and come back later; water is fixed and unlimited;
   predators lurk near food and chase on sight; and the agent carries hunger, thirst,
   fatigue and health that it must all keep above half-full to earn reward.
2. **An analysis framework** (their Table 1) mapping six questions — what goals? how
   far back does memory reach? how far ahead does it plan? how does it move? is the
   network the right size? what is encoded? — onto concrete diagnostic methods.
3. **The empirical finding**: an ordinary model-free recurrent agent, given no world
   model and no memory module beyond a 512-unit gated recurrent network, produces
   behaviour and internal structure that look like planning.
4. **Released code** for environment, agents and analyses.

**What they found, in plain terms.**

- *Exploration has shape.* Trained agents leave the arena centre in outward-expanding,
  rotating loops that progressively cover the arena — a pattern the authors show
  side-by-side with published search trajectories of displaced desert ants and of bees
  on their first flights from the hive.
- *Then behaviour switches phase.* After mapping the arena, agents stop wandering and
  start making direct trips to remembered food patches, while still exploring enough
  to keep finding shortcuts and predator-free routes. This mirrors a well-known
  transition in mice learning a labyrinth.
- *Patch choice is multi-factor and sensible.* Fitting a regression to ~8,000 revisit
  decisions, agents prefer patches they have eaten from *less*, patches where they saw
  *more* food, patches visited *more recently*, and patches where they were attacked
  *less*. Proximity to water did not matter.
- *Movement decomposes into modes.* Unsupervised clustering of step size and turning
  angle yields three movement states — short-, mid-, long-range — and the mid-range
  one co-occurs with predators while the short-range one co-occurs with eating.
- *Skills arrive in stages.* Early in training agents sit near the start point
  "fishing"; then, around 20,000 training iterations, several behavioural measures
  jump at once — longer trips, better food/water trade-offs, tool crafting, predator
  defence. None of this is visible in the reward curve, which rises smoothly.
- *The memory carries a map-like signal.* A plain linear readout (ridge regression)
  from the 512-unit recurrent state recovers where the agent was, or will be, 50–100
  steps away in either direction — but only in *world-anchored* coordinates (relative
  to the arena origin), not in body-centred coordinates, which stay at chance.
- *Architecture choices change interpretability, not just performance.* Cutting 90% of
  the recurrent weights leaves performance intact and makes the spatial code *easier*
  to read out. Adding a side-task that asks the network to report its own position
  improves both performance in big arenas and the readability of the code.

**Initial takeaway.** Two claims, of unequal strength. The strong and well-supported
one is methodological: *behaviour-first analysis reveals structure that reward curves
hide*, and the authors demonstrate this repeatedly. The weaker one is scientific:
*model-free RNN agents plan through emergent dynamics*. The behavioural half of that
holds up reasonably; the neural half depends on a supervised auxiliary objective
whose removal destroys the effect, which the paper reports honestly in its results
(their Figure 17) but does not carry into its abstract.

---

### 5.3 Phase 2 — Graduate-Level Deep Dive

#### 5.3.1 The ForageWorld environment as a formal object

**Arena.** Each episode instantiates a fixed-size grid

$$
\mathcal{G} = \{0,\dots,95\}^2, \qquad |\mathcal{G}| = 96 \times 96 = 9{,}216 \text{ cells},
$$

procedurally generated: cow spawn points and lake regions are placed in patches drawn
from **Perlin noise**, and obstacles are inserted to break line of sight. The agent is
initialised at the arena centre, $(x_0,y_0) = (48,48)$ **[reviewer: the paper states
"center of the arena" but never writes the coordinate]**. This fixed origin is
load-bearing for every allocentric analysis downstream — see §5.3.5.

**Resource dynamics.** Let $\mathcal{C}_t \subset \mathcal{G}$ be the set of occupied
cow cells at time $t$, with a hard arena-wide cap

$$
|\mathcal{C}_t| \le N_{\max} = 108 .
$$

Cows spawn at fixed points and perform a diffusion (random walk) between steps;
consumption removes a cow. The behavioural consequence the authors want is a
**local, temporary depletion**: eating out a patch drops the local density, which
recovers only as cows diffuse back in, so the optimal policy must *leave and return*.
$N_{\max}$ was selected by sweep over $\{48, 72, 108\}$ specifically to tune the
balance between patch-leaving and revisitation in competent agents. Water sources
(lakes) are fixed and unlimited — hence the deliberately asymmetric resource economy
(one renewable-but-depleting, one inexhaustible-but-immobile).

**Threats.** Melee and ranged predators spawn intermittently near food patches and
pursue the agent when it is within line of sight. Spawn rate is modulated by a
**light level** variable (more predators at night). Predators can damage the agent and
can themselves be killed if the agent crafts a weapon — combat is possible but, as the
predator-blind ablation shows, not necessary.

**Physiological state and reward.** The agent maintains four scalars — health, food,
drink, energy. Fatigue recovers only by sleeping, which is gated on
$\text{energy} < 0.5\,\text{energy}_{\max}$, immobilises the agent, and gradually
restores energy and health. Starvation and extreme thirst decrement health; the
episode terminates when health reaches zero. The reward is a **thresholded homeostatic
maintenance** signal (their Eq. 1):

$$
R(s_t) \;=\; 0.1\Big[\,1 \;+\; \operatorname{sign}\!\big(\text{health}_t - 5\big)
\;+\; \operatorname{sign}\!\big(\text{food}_t - 5\big)
\;+\; \operatorname{sign}\!\big(\text{drink}_t - 5\big)
\;+\; \operatorname{sign}\!\big(\text{energy}_t - 5\big)\Big].
$$

**Reading the reward — [reviewer].** Each $\operatorname{sign}(\cdot) \in \{-1,0,+1\}$,
so the bracket ranges over the integers $[1-4,\,1+4] = [-3,\,5]$ and

$$
R(s_t) \in \{-0.3,\,-0.2,\,\dots,\,0.5\},
$$

a per-step signal, positive iff at least three of the four variables sit strictly
above their half-max threshold of 5. Three design consequences follow. (i) The reward
is **saturating**: pushing food from 6 to 9 earns nothing, so there is no incentive to
over-graze — precisely the property the authors want, and precisely the property that
an "accumulate as much as possible" reward lacks. (ii) The reward is **piecewise
constant**, hence gives zero gradient information about *how far* a variable is from
its threshold; all the shaping comes from the discount and the value function. (iii)
Because $R$ is bounded and mostly positive for a competent agent, **return is
approximately proportional to episode length**, which the authors state explicitly
("long survival times (directly proportional to return)"). The authors cite EVAAA
(Lee et al., NeurIPS 2025 Datasets & Benchmarks) — this repository's sibling platform
— as the justification for this homeostatic-buffer reward design over a maximisation
reward.

**Episode structure.** Maximum length $T_{\max} = 100{,}000$ steps; termination is
almost always by $\text{health} = 0$ rather than by the cap, even for competent
agents. Decoding analyses restrict to steps $1000 \le t \le 6000$ per episode, which
bounds the usable episode length in practice.

**Action space.** A subset of the Craftax action space, with actions irrelevant to
foraging disabled (`Use full action space: False`).

#### 5.3.2 Observation and state

At each step the agent receives

$$
o_t \;=\; \big[\; v_t,\;\; \iota_t,\;\; \psi_t \;\big],
$$

where $v_t$ is a **$9 \times 11$ egocentric tile window** centred on the agent
(99 cells, i.e. $99/9216 \approx 1.07\%$ of the arena visible at once **[reviewer]**),
$\iota_t$ is an inventory vector, and $\psi_t$ collects the internal physiological
variables (health, food, drink, energy, and the associated timers). A feedforward
encoder $\phi$ maps this to the recurrent input:

$$
x_t = \phi(o_t).
$$

An ablation replaces the $360^\circ$ window with a **front-facing** field of view
(only the forward grid rows), which makes the task more information-seeking and more
animal-like; this is off by default (`Directional Vision: False`).

**Logging schema.** The paper's Table 2 defines the per-step log, and this is the part
of the contribution most worth copying. It records: `Action`; the four physiological
scalars plus `Done`, `Is Sleeping`, `Is Resting`; `Player Position` (absolute, origin
at top-left) and `Delta X & Y` (relative to episode start); the four countdown timers
`Recover`/`Hunger`/`Thirst`/`Fatigue`; `Light Level`; per-class L1 distances and
on-screen flags and counts for melee predators, ranged predators, and cows;
`Predicted Delta X & Y` (the auxiliary head's own position estimate);
`Num Monsters Killed`, `Has Sword`, `Has Pick`, `Held Iron`; and — critically for the
neural analyses — the **full recurrent hidden state** plus the policy's `Value`,
`Entropy` and `Log Probability` of the selected action, and an `Episode ID`.

**[reviewer] — the design principle behind that table.** Every logged field is either
(a) a ground-truth quantity a decoder can be *trained to predict*, (b) a regressor for
a choice model, or (c) an internal quantity from which those are decoded. There are no
fields logged "just in case" that do not appear in an analysis. That is the actual
transferable lesson: the log schema is designed backwards from the analysis
framework, not forwards from what the simulator happens to expose.

#### 5.3.3 Agent architecture and objective

**Recurrent core.** A single-layer GRU with $|h| = 512$ units, shared by actor and
critic:

$$
h_t = \mathrm{GRU}\big(x_t,\, h_{t-1}\big), \qquad h_t \in \mathbb{R}^{512},
$$

expanding to the standard Cho et al. gating equations:

$$
z_t = \sigma\!\big(W_z x_t + U_z h_{t-1} + b_z\big)
$$
$$
r_t = \sigma\!\big(W_r x_t + U_r h_{t-1} + b_r\big)
$$
$$
\tilde{h}_t = \tanh\!\big(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h\big)
$$
$$
h_t = (1 - z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t
$$

**Derivation — why a GRU can hold a 100-step memory [reviewer].** Consider a single
unit whose update gate is approximately constant, $z_t \approx z$, and whose candidate
drive is zero. Then $h_t = (1-z)\,h_{t-1}$, so $h_t = (1-z)^t h_0$, and writing
$(1-z)^t = e^{t\ln(1-z)} = e^{-t/\tau}$ gives a decay time constant

$$
\tau \;=\; -\frac{1}{\ln(1-z)} \;\approx\; \frac{1}{z} \quad (z \ll 1).
$$

To retain information over the $\sim$100-step horizon the decoding analysis reports,
the network needs a subpopulation operating at $z \lesssim 10^{-2}$. This is the
mechanistic reason recurrence — rather than a feedforward encoder — is load-bearing
here, and it predicts that the memory-carrying units should be a *minority* of slow
units rather than the whole population. The paper does not measure the gate
distribution; doing so would be a cheap and direct test of this account.

**Heads.** Policy, value, and (optionally) a path-integration head all read the same
$h_t$:

$$
\pi_\theta(a_t \mid o_{\le t}) = \mathrm{softmax}\big(W_\pi h_t + b_\pi\big), \qquad
V_\theta(o_{\le t}) = w_V^\top h_t + b_V, \qquad
\hat{p}_t = W_p h_t + b_p \in \mathbb{R}^2 .
$$

Total trainable parameters $\approx 6.5\times10^6$; activation $\tanh$; 512 units per
layer.

**PPO objective** (their Eq. 3):

$$
L^{\mathrm{clip}}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\;
\mathrm{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat{A}_t\big)\Big],
\qquad
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)} ,
$$

with $\epsilon = 0.2$. Advantages come from GAE:

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l},
\qquad
\delta_t = r_t + \gamma V(h_{t+1}) - V(h_t),
$$

with $\gamma = 0.99$, $\lambda_{\mathrm{GAE}} = 0.8$.

**Auxiliary path-integration objective** (their Eq. 2). The position head regresses
the agent's displacement from its episode-start origin:

$$
\mathcal{L}_{\mathrm{aux}} = \mathbb{E}_t\Big[\;\big\lVert \hat{p}_t - p_t \big\rVert_2^2\;\Big],
\qquad p_t = (x_t - x_0,\; y_t - y_0).
$$

The composite loss, with the paper's coefficients, is **[reviewer — assembled from
§3.2, §A.6 and Table 3; the paper never writes the total loss in one place]**:

$$
\mathcal{L}(\theta) = -L^{\mathrm{clip}}(\theta)
\;+\; W_V \,\mathbb{E}_t\big[(V_\theta(h_t) - \hat{R}_t)^2\big]
\;-\; W_{\mathrm{ent}}\,\mathbb{E}_t\big[\mathcal{H}[\pi_\theta(\cdot\mid h_t)]\big]
\;+\; W_{\mathrm{aux}}\,\mathcal{L}_{\mathrm{aux}} ,
$$

with $W_V = 0.5$, $W_{\mathrm{ent}} = 0.01$, $W_{\mathrm{aux}} = 0.025$ (selected by
sweep over $\{0.01, 0.025, 0.1, 1.0\}$, "balancing performance and position encoding
quality"). Gradient norms clipped to 1.0; Adam with default momenta.

**Two timescales worth separating [reviewer].** The effective credit-assignment
horizon of GAE and the effective horizon of the value function are *not* the same
number here:

$$
H_{\mathrm{GAE}} = \frac{1}{1-\gamma\lambda} = \frac{1}{1-0.99\times0.8} \approx 4.8 \text{ steps},
\qquad
H_{V} = \frac{1}{1-\gamma} = 100 \text{ steps}.
$$

The reported "planning horizon" of 50–100 steps coincides almost exactly with $H_V$,
not with $H_{\mathrm{GAE}}$. This is either a satisfying consistency (the network
represents the future over exactly the horizon the value function integrates) or a
confound (the decodable horizon is *set by* $\gamma$ and has nothing to do with
planning). §5.3.7 argues it is currently impossible to tell from this paper alone, and
§5.4 argues this repository can settle it cheaply.

**Throughput arithmetic [reviewer].** 1024 parallel environments $\times$ 64 steps per
PPO iteration $= 65{,}536$ transitions per iteration; 4 epochs $\times$ 8 minibatches
gives a minibatch of $65{,}536/8 = 8{,}192$, matching the stated figure. Total budget
$3\times10^9$ steps $\Rightarrow \approx 45{,}776$ PPO iterations. Logging every 2048
iterations $\Rightarrow \approx 134\times10^6$ steps between behavioural samples, i.e.
**the entire training-dynamics analysis rests on ~22 sample points**. One GPU (A100 or
H100) per run, 24–48 h wall clock, 5 seeds.

**Sparsification.** Magnitude pruning via JaxPruner, applied at iteration 20,000, to a
target sparsity of 90% across model layers, motivated by the 70–94% sparsity range
reported for biological connectomes. Formally, for each weight matrix $W$ a mask

$$
M_{ij} = \mathbb{1}\Big[\,|W_{ij}| \ge \tau_{0.9}(|W|)\,\Big],
\qquad W \leftarrow M \odot W,
$$

with $\tau_{0.9}$ the 90th percentile of the magnitude distribution. Magnitude pruning
was the only JaxPruner method that did not severely degrade performance in their
testing.

#### 5.3.4 Behavioural analysis methods

The paper's Table 1 is the organising device — a mapping from question to instrument:

| Analysis target | Key question | Methods used |
|---|---|---|
| Goal inference | What goals is the agent pursuing, and how are they shaped by experience? | Decision GLMs, in-context learning study, behaviour phase segmentation |
| Memory span | How far back in time can the agent remember? | RNN state decoding, memory-module ablations |
| Planning horizon | How far ahead does the agent plan? | Future decoding, auxiliary-loss analysis for predictive objectives |
| Spatial structure | How does the agent move through the environment? | Position-occupancy entropy, tracking revisitation, path analysis |
| Network capacity | Is the model too large or too small for the task difficulty? | Network size / pruning sweeps, comparing performance curves |
| Representations | What information is encoded in the agent's internal states? | Decoding, GLMs, encoding profiles of task variables |

**(a) Path analysis.** Qualitative but systematic: for each episode, plot (i) the
early exploratory segment, (ii) a segment sampled during the exploration-to-revisit
transition, and (iii) the full trajectory with revisited patch regions highlighted.
Paths 1–4 in the paper's Figure 1C are *consecutive* segments of one episode, with
Path 1 starting at $t=0$. The comparison target is published animal search data:
displaced desert ants (Wehner & Srinivasan 1981) and first, second and third bee
departures from the hive (Osborne et al. 2013), both showing azimuthally rotating,
outward-expanding loops.

**(b) Position-occupancy entropy.** The paper names the statistic but does not write
it. The standard form **[reviewer]** is, for visit counts $n_c$ over spatial bins
$c$ and $N = \sum_c n_c$,

$$
H_{\mathrm{occ}} = -\sum_{c} \frac{n_c}{N}\,\log \frac{n_c}{N},
\qquad 0 \le H_{\mathrm{occ}} \le \log |\mathcal{B}| ,
$$

maximised by uniform coverage of the bin set $\mathcal{B}$. With $|\mathcal{B}| = 9216$
this ceiling is $\log 9216 \approx 9.13$ nats, so on this arena the statistic is far
from saturating and genuinely discriminates broad from narrow foragers.

**(c) Distance-from-origin, split early vs late.** Mean $\lVert p_t \rVert$ computed
separately over $t < 1500$ and $t \ge 1500$ within an episode, which operationalises
the exploration/revisitation phase split with a fixed cut point.

**(d) Angular orientation variance.** Variance of heading computed over 250-timestep
windows — a coarse measure of how directed movement is.

**(e) Spatial uncertainty.** The auxiliary head's own error, normalised by distance
from origin:

$$
u_t \;=\; \frac{\lVert \hat{p}_t - p_t\rVert}{\lVert p_t \rVert} \quad \textbf{[reviewer — the paper says "normalized by distance from origin" without writing the ratio]}.
$$

This quantity does double duty: as a training-dynamics metric, and as a *regressor*
in the patch-choice model below. Note it exists only for agents trained with the
auxiliary objective — in the ablation the corresponding panel is simply absent.

**(f) Behavioural phase segmentation (their §B.2).** Unsupervised, two-stage:

1. **Movement-state clustering.** The `bayesmove` package (Cullen et al. 2022, a
   nonparametric Bayesian mixture over movement features) is applied to two locomotion
   features — **turning angle** and **step size** — computed over a **7-timestep
   moving window**, yielding $K = 3$ latent movement states. Post-hoc they read as
   short-range (state 3), mid-range (state 1) and long-range (state 2) navigation.
2. **Transition prediction.** Conditional inference trees (`ctree`, Hothorn et al.)
   are fit to predict transitions between those states from task variables. Top
   predictors: hunger (`food`), positional uncertainty, predator presence
   (`enemy_present`), nearby cow density (`num_passives_nearby`).

The finding is that state 1 co-occurs with predator events (evasive/scanning) and
state 3 clusters near food and eat actions — i.e. the agent switches movement regime
in response to internal drive and external context **with frozen weights, on held-out
arenas**, which is the control that makes this an in-context rather than a learning
effect.

**(g) Patch-revisitation decision GLM (their Figure 3 and §B.4).** The paper's single
most transferable analysis. Construction:

- **Unit of analysis:** a revisitation *decision*, defined at 50 timesteps **before**
  each patch-eat event.
- **Response:** binary — was this patch the one chosen ($y = 1$) or not ($y = 0$).
- **Regressors**, each a historical average over the agent's experience of that patch:
  `EatRate` (prior eat actions there), `DrinkRate` (water proximity), `PredRate`
  (predator encounters there), `Recency` (how recently visited), `Dwelltime`,
  `CowCount` (cows observed there), `Uncertainty` (prior position-prediction error
  there).
- **Structure:** one model fit jointly across 5 PPO-RNN agents spanning **7,978**
  revisitation decisions, with agent identity as a **fixed effect**; implemented with
  `statsmodels.formula.api`.
- **Diagnostics:** variance inflation factors reported, all $\mathrm{VIF} < 10$.

Writing it out **[reviewer]**, the model is a logistic regression

$$
\log\frac{\Pr(y_{ij}=1)}{1-\Pr(y_{ij}=1)}
= \beta_0 + \sum_{k} \beta_k\, X_{ijk} + \alpha_{a(i)},
$$

for decision $i$, candidate patch $j$, regressor $k$, and agent-identity fixed effect
$\alpha_{a(i)}$; VIF for regressor $k$ is $\mathrm{VIF}_k = 1/(1-R_k^2)$ where $R_k^2$
is from regressing $X_k$ on the remaining regressors.

**Results:** significant preference for patches with *fewer* prior eats, *fewer*
predator encounters, *more recent* visits, *more* observed cows, and *higher* prior
position-prediction error; a mild preference for longer dwell time; and no effect of
water proximity.

**Interpreting the sign pattern [reviewer].** Two of these coefficients point in
opposite directions with respect to optimal-foraging logic. Preferring patches with
fewer prior eats is depletion-consistent (avoid what you exhausted). Preferring
*more recently visited* patches is depletion-*inconsistent* — a return-maximiser
should prefer the patch it has left alone longest, giving cows time to diffuse back.
The recency effect is more parsimoniously read as **memory decay** (the agent goes
where its representation is freshest) than as resource-return maximisation. The paper
does not reconcile these.

**(h) Policy-entropy dynamics.** Mean policy entropy
$\mathcal{H}[\pi] = -\sum_a \pi(a)\log\pi(a)$ and mean log-probability of the selected
action, tracked across training. Entropy falls as the policy sharpens, then rises
slightly late in training — which the authors attribute to the entropy bonus term
letting the agent find high-return *and* high-entropy solutions.

#### 5.3.5 Neural analysis methods

**(a) Allocentric / egocentric displacement decoding.** The central method. Record
hidden states $h_t \in \mathbb{R}^{512}$ and ground-truth positions $(x_t,y_t)$
throughout each episode. For each temporal offset $\Delta t$ (positive = future,
negative = past), define the target

$$
Y_{t+\Delta t} = \big(x_{t+\Delta t} - x_t,\; y_{t+\Delta t} - y_t\big) \in \mathbb{R}^2
$$

and fit a **separate linear decoder per offset**

$$
f_{\Delta t}(h_t) = A h_t + b, \qquad A \in \mathbb{R}^{2\times 512},\; b \in \mathbb{R}^2,
$$

by ridge regression (their §C.3):

$$
\mathcal{L}_{\Delta t}(f) = \sum_{i=1}^{N}\Big(Y^{\,i}_{t+\Delta t} - f\big(h^{\,i}_t\big)\Big)^2
\;+\; \alpha \lVert f \rVert^2_{K}.
$$

**Derivation of the closed form [reviewer].** Stack the $N$ hidden states as rows of
$H \in \mathbb{R}^{N\times 513}$ (last column of ones absorbing the bias) and the
targets as $Y \in \mathbb{R}^{N\times 2}$, and write $W = \begin{bmatrix}A^\top\\ b^\top\end{bmatrix} \in \mathbb{R}^{513\times 2}$. Then

$$
\mathcal{L}(W) = \lVert Y - HW\rVert_F^2 + \alpha\lVert W\rVert_F^2 .
$$

Differentiating,

$$
\frac{\partial \mathcal{L}}{\partial W} = -2H^\top\big(Y - HW\big) + 2\alpha W \;\stackrel{!}{=}\; 0
\;\;\Longrightarrow\;\; \big(H^\top H + \alpha I\big)W = H^\top Y
\;\;\Longrightarrow\;\; W = \big(H^\top H + \alpha I\big)^{-1}H^\top Y .
$$

The $\alpha I$ term is what makes this well-posed at all: the authors note that some
GRU units are **inactive for entire episodes**, so $H^\top H$ is rank-deficient and the
ordinary least-squares normal equations are singular. Forming $H^\top H$ costs
$O(Np^2)$ with $p = 512$ — the complexity the authors quote — and the solve costs
$O(p^3)$, negligible by comparison for large $N$.

**Coordinate frames.** Two decoder families are fit — **allocentric** (displacement
relative to the episode origin) and **egocentric** (relative to body orientation) — as
a matched contrast. Only the allocentric one works.

**Train/test protocol.** Within each episode, train on the **first 75%** of timesteps
and evaluate on the **final 25%**, explicitly to defeat leakage from the temporal
autocorrelation of $h_t$. Decoders are trained across multiple episodes; the *number*
of training arenas per decoder is varied as a control to show the decoder is not
exploiting arena-specific cues. Accuracy is reported as RMSE against an
**average-displacement chance baseline**. All analyses run on held-out test arenas
with frozen weights.

**Results.** (i) Egocentric decoding stays at chance across models. (ii) Allocentric
displacement decodes above chance for $|\Delta t| \lesssim 50$–$100$ steps in both
directions. (iii) Decoding accuracy improves over training. (iv) A *single* decoder
generalises across episodes and arenas — which the authors attribute to the invariant
$96\times96$ layout and global orientation providing a stable, compass-like reference
frame.

**(b) Coefficient structure and functional modularity (their §C.2).** Take the ridge
weight matrices $A^{(\Delta t)}$ for past and future offsets, align them by neuron
index, and compare which units carry weight. Both sparse and dense agents use
overlapping populations for past and future, but the overlap **shrinks as $|\Delta t|$
grows**, and sparse (pruned) agents show cleaner separation. **[reviewer]** The paper
reports this visually; the natural scalar summary it does not compute is a normalised
overlap such as

$$
\rho(\Delta t) = \frac{\big\langle |A^{(-\Delta t)}|,\, |A^{(+\Delta t)}| \big\rangle}
{\lVert A^{(-\Delta t)}\rVert\,\lVert A^{(+\Delta t)}\rVert} ,
$$

which would turn "the overlap diminishes" into a testable curve.

**(c) Single-neuron position-encoding GLM (their Appendix D).** The complementary
*encoding* direction: predict each unit's activity **from** position, rather than
position from the population.

- Coarse-grain the arena into $14\times14 = 196$ position bins (each bin
  $\approx 6.86 \times 6.86$ cells **[reviewer]**), and one-hot encode occupancy:
  $z_t \in \{0,1\}^{196}$, $z_{t,k} = \mathbb{1}[\,b(x_t,y_t) = k\,]$.
- Normalise each unit's activity to $[0,1]$ after shifting the minimum to zero:

$$
\tilde{a}^{(n)}_t = \frac{a^{(n)}_t - \min_t a^{(n)}_t}{\max_t a^{(n)}_t - \min_t a^{(n)}_t}.
$$

- Fit **one GLM per unit** with NeMoS (Balzani et al.), pooling 5–10 episodes per fit
  to avoid over-fitting a single arena layout:

$$
\mathbb{E}\big[\tilde{a}^{(n)}_t \mid z_t\big] \;=\; g^{-1}\!\Big(\beta^{(n)}_0 + \sum_{k=1}^{196}\beta^{(n)}_k z_{t,k}\Big).
$$

- Train on the first 70% of each episode, test on the last 30%; report the train/test
  log-likelihood gap per unit.

**Results.** Roughly **100 of 512 units** (typically 60–120 per run) are well fit by
position alone. Clustering the coefficient vectors with $k$-means ($k=5$, retaining
the three largest-mean clusters) and mapping coefficients back to arena bins gives two
findings: the **number** of position-sensitive units per bin and the **average
coefficient magnitude** both **increase with radial distance from the origin**. The
authors read this as a *distance-accumulation circuit* — a signal ramping with
displacement from home that pressures the agent to return, offering a mechanistic
account of the observed periodic origin-revisiting. Critically, this radial ramp
appears in **all** conditions tested: PPO-RNN with path integration, PPO-RNN
*without* it, and PQN-RNN — "suggesting it is fundamental to the task solution".

**[reviewer] — an unresolved internal tension.** Removing the auxiliary objective
drives *population ridge decoding of displacement* to chance (their Figure 17), yet
*per-unit position tuning* survives it (their Figures 25–26). These are not
contradictory, but the paper leaves the reconciliation to the reader. The most likely
account: the surviving single-unit tuning is essentially **radial** — a scalar
"how far from home am I" accumulator — and a scalar cannot support linear readout of a
2-D displacement vector. If so, the aux objective is what converts a *distance*
signal into a *vector* signal, which is a sharper and more interesting claim than
either of the ones the paper makes. Testing it needs only a re-analysis of existing
data: fit per-unit GLMs with $\lVert p_t\rVert$ alone as the regressor and compare
explained deviance against the 196-bin model, with and without the aux loss.

**[reviewer] — a reproducibility gap.** The link function $g$ and noise family are
never stated. NeMoS's default family is Poisson (log or soft-plus link), which is
appropriate for spike counts and **mis-specified** for continuous $\tanh$-derived GRU
activations rescaled to $[0,1]$. Either a Gaussian-identity GLM was used (in which
case the "GLM" is ordinary regression and should say so) or a Poisson likelihood was
applied to non-count data (in which case the log-likelihood comparisons in their
Figure 20 are hard to interpret). Anyone porting this analysis must pin the family
explicitly.

#### 5.3.6 Ablations, sweeps and the alternative algorithm

| Manipulation | Effect on performance | Effect on internal structure |
|---|---|---|
| Replace 512-unit RNN with feedforward | Large impairment; agents loop locally, survive far fewer steps | n/a |
| Downsize RNN to 64 units (550k params) | Large impairment | n/a |
| 128-unit network **+ pruning** | Underperforms — "high capacity is needed to support sparsity" | n/a |
| 90% magnitude pruning at 512 units | No degradation | **Improves** allocentric decoding; cleaner past/future separation in decoder weights |
| Remove auxiliary path-integration loss | Reduces performance in **large** arenas only; degrades emergence of predator killing | **Destroys** above-chance allocentric decoding; radial per-unit tuning survives |
| Front-facing FOV instead of $360^\circ$ | Improves *early* learning; equal final performance | Earlier onset of long-range exploration; denser local search early |
| Predator-blind (agent cannot damage predators) | Comparable survival and movement | Evasion, not combat, is the primary learned strategy |
| Swap PPO-GRU $\to$ PQN-LSTM | Comparable *for the runs that converged*; **only ~30% converged** | Similar learning histories and pathing; markedly lower predator-killing rate |

**On the PQN comparison.** PQN (parallelised Q-networks, Gallici et al.) was given
$1.2\times10^{10}$ timesteps — **4$\times$ the PPO budget** — across two exploration
configurations, plus sweeps over $\epsilon$-finish $\{0.005,0.01,0.05,0.1\}$ and
$\epsilon$-decay $\{0.1,1.0,2.0\}$ and total-decay $\{1,6,12\}\times10^9$. The
authors' own diagnosis of the failures is premature convergence to simple,
reliable-but-suboptimal behaviour. Their behavioural conclusion is the interesting
part: at matched *performance*, the two algorithms adopt **different strategies** —
PPO converges to predator fighting, PQN to predator evasion. That is exactly the kind
of finding reward curves cannot express, and it is the paper's best single
demonstration of its own thesis.

#### 5.3.7 What the evidence licenses — critical assessment

**Strengths.**

1. **The thesis is demonstrated, not just asserted.** The PPO-vs-PQN strategy
   divergence at matched return, the staged skill emergence invisible in a smooth
   reward curve, and the pruning result (performance flat, interpretability up) are
   three independent existence proofs that behaviour-first analysis sees what reward
   curves cannot. That is the contribution, and it lands.
2. **Methods are imported honestly, with citations to their originating literature**
   (`bayesmove`, `ctree`, NeMoS, ridge decoding, occupancy entropy), rather than
   re-invented — which is what makes the toolkit reusable rather than bespoke.
3. **The negative controls that exist are the right ones.** Egocentric decoding at
   chance is a genuine specificity control for the allocentric result; varying the
   number of training arenas per decoder controls arena-specific cues; frozen weights
   on held-out arenas separate in-context adaptation from learning; the within-episode
   75/25 split addresses temporal-autocorrelation leakage.
4. **Ablation breadth is unusual** — capacity, recurrence, sparsity, auxiliary
   objective, sensory field of view, combat affordance, and a second RL algorithm.
5. **Environment design is thoughtful where it matters.** Cow cap chosen by sweep to
   *induce* patch-leaving; map size chosen by sweep to maximise memory demand; reward
   deliberately saturating to prevent over-consumption.
6. **Reporting hygiene:** 5 seeds, 95% CIs throughout, VIF diagnostics on the GLM,
   compute disclosed, code released, complete NeurIPS checklist, limitations section
   that names real limitations.

**Weaknesses.**

1. **The headline over-claims relative to the paper's own ablation.** The abstract
   says model-free RNN agents "exhibit structured, planning-like behavior purely
   through emergent dynamics". The principal *neural* evidence for planning —
   allocentric future decoding — goes to chance without $\mathcal{L}_{\mathrm{aux}}$,
   a **supervised regression onto ground-truth position**. A representation induced by
   explicit supervision is not "emergent from model-free RL". The behavioural evidence
   (revisitation, phase structure) is less compromised, but the paper's rhetorical
   weight sits on the neural half.
2. **No causal test of the putative planning representation.** Everything is
   decoding, i.e. correlational. The neuroscience-standard next move — perturb the
   decoded subspace and show the future trajectory shifts accordingly — is available
   in simulation at zero experimental cost and is not done. Without it, "the agent
   plans" and "the agent's state happens to correlate with its future" are
   observationally identical.
3. **The chance baseline for future decoding is too weak.** "Average displacement per
   timestep" does not control for the obvious deflationary explanation: if the agent
   moves smoothly and near-ballistically, its position 50 steps hence is close to a
   deterministic function of its *current position and heading*, both of which are
   trivially recoverable from $h_t$. The needed control is a **behaviour-matched
   surrogate** — decode $Y_{t+\Delta t}$ from $(p_t, v_t)$ alone with no hidden state,
   and show $h_t$ beats it — plus a shuffled-$h$ control. Neither appears.
4. **The reported horizon coincides with $1/(1-\gamma) = 100$ steps.** As derived in
   §5.3.3, the value function's own integration horizon numerically matches the
   reported planning horizon. Nothing in the paper distinguishes "the network
   represents a 100-step plan" from "the network represents whatever the discount
   factor makes it represent". A $\gamma$-sweep would settle it and is not run.
5. **The PPO-vs-PQN comparison conditions on success.** Comparing PPO to the ~30% of
   PQN runs that converged, at 4$\times$ the timestep budget, is not an algorithm
   comparison; it is a comparison of PPO to the best tail of PQN. The *behavioural*
   contrast (fight vs. flee) is still interesting, but the "performed comparably"
   framing should carry the selection caveat more prominently than a subordinate
   clause.
6. **The revisitation GLM's statistics are under-specified.** 7,978 decisions from 5
   agents are pooled with agent identity as a fixed effect, but nothing indicates
   clustered or robust standard errors. Decisions within an agent — indeed within an
   episode — are strongly dependent, so nominal CIs are very likely **too narrow** and
   the significance stars correspondingly optimistic. The **choice set** is also
   unspecified: which patches counted as available alternatives at each decision, and
   how "patch" was defined spatially, determines what the coefficients mean.
7. **The "abrupt shift at ~20,000 iterations" has two problems.** First, resolution:
   behavioural metrics are logged every 2048 iterations, so "abrupt" is measured at a
   $\sim$134-million-step grain and could be any transition faster than that. Second,
   and more seriously, **the prune step is also 20,000** (their Table 3). The paper
   never states whether the agents in the training-dynamics figure are pruned; if they
   are, the "emergence of higher-level strategies" coincides exactly with a
   discontinuous intervention on the network. This needs one sentence of clarification
   and currently does not have it.
8. **Internal inconsistencies.** The learning rate is 0.00025 in §A.5 and 0.0002 in
   Table 3. §4.3 says "PPO seems to converge to a predator fighting strategy" while
   their Figure 12 shows that removing the ability to damage predators changes nothing
   — both can be true (learned but not needed), but the text does not reconcile them.
   The stated eat rate of 0.011 per step is "approximately once every 95 steps"; it is
   once every $\approx 91$ steps **[reviewer]**.
9. **Table 1 is advertised as a reusable framework but is a pointer table.** It names
   questions and method families. It contains no operational definitions, no
   thresholds for "the effect is present", no sample-size guidance, no failure modes.
   A genuinely reusable protocol needs those — as the paper's own analyses
   demonstrate, since almost every method in them required a design decision (window
   length, bin count, offset grid, choice-set construction) that Table 1 does not
   record.
10. **No model-based comparison.** The claim is framed as "without world models", but
    no world-model agent is trained in ForageWorld, so there is no calibration for how
    much better the behavioural and neural signatures would look with one.
11. **Single environment.** The framework's generality is asserted; only ForageWorld
    is tested.

**Net assessment.** As a *methods and position* paper it is strong and genuinely
useful — the environment is well designed, the logging schema is the best transferable
artefact in it, the analysis pipeline is real and released, and the demonstration that
equal-return agents can differ in strategy is a clean argument for the thesis. As an
*empirical claim about implicit planning* it is over-stated: the neural evidence
depends on a supervised auxiliary objective, is purely correlational, lacks a
behaviour-matched surrogate control, and reports a horizon numerically equal to the
discount horizon. Cite it for the framework, the environment design and the logging
schema; cite the planning claim only with the auxiliary-loss caveat attached.

---

### 5.4 Relevance to this project

This repository trains model-free recurrent-PPO agents in a partially observable
grid-world with predators, depleting-and-regenerating food, bushes that hide the
agent, and interoceptive state (satiation, delayed nociception, hidden injury and
nutrition). Its evaluation principle is **survival steps, never cumulative reward**,
and it already owns a step-level qualitative microscope
(`scripts/eval/trajectory_story.py`, exposed as the `/trajectory-story` skill) plus a
pre-registered behaviour-measure toolkit (M1/M2/M5 online, M7 motif clustering
offline). The paper is therefore the closest methodological neighbour in the whole
reference library — and there is a direct citation link in the other direction: the
paper justifies its homeostatic reward design by citing **EVAAA (Lee et al., NeurIPS
2025 D&B)**, this project's sibling platform.

#### 5.4.1 Directly portable methods, and what data each needs

Grounded in `src/utils/eval_recording.py` (the recording payload),
`scripts/eval/eval_rollout.py` (what the rollout computes), and
`scripts/eval/trajectory_story.py`.

**What the recorder captures today.** `EpisodeRecorder.append` stores, per step:
`agent_pos`, `satiation`, `nutrition`, `injury_level`, `rest_streak`, `res_pos`,
`res_active`, `animal_pos`, `obs_pos`, plus the noisy observation `obs`, the
noise-free `true_obs`, the action index, and the reward.

**What it does not capture — and this is the important part.** The batched rollout
scan computes, at every step,

```python
logits, _value, h_new, _mod_info = model(v_obs(state, params), h)
```

and **discards** three of those four returns. So the GRU hidden state, the value
estimate, the policy logits (hence entropy and log-probability of the selected
action), and any modulation diagnostics are all *already computed inside the eval
loop* and simply not written to the recording. Every neural analysis in this paper is
blocked on exactly that one gap, and nothing else.

| Paper method | Portable? | Data required | Already captured? |
|---|---|---|---|
| Path / trajectory analysis, revisitation highlighting | Yes, mechanically | `agent_pos` time series | ✅ |
| Distance-from-origin, early vs late split | Yes, with a redefinition (see §5.4.2) | `agent_pos`, episode-start cell | ✅ (`snapshots[0]['agent_pos']`) |
| Position-occupancy entropy | Yes, but near-ceiling (see §5.4.2) | `agent_pos` | ✅ |
| Angular / heading variance | Yes, coarsely | consecutive `agent_pos` | ✅ |
| Movement-state segmentation (step size + turn angle, 7-step window) | **Superseded** — the project's M7 motif clustering already occupies this niche with a richer 10-feature vector and the same $K=7$ window | window features | ✅ (M7) |
| Conditional inference trees predicting state transitions | **Yes — highest-value cheap import.** Fit `ctree` on M7 motif labels with physiological + threat covariates | motif labels + per-step covariates (`nutrition`, `injury_level`, `satiation`, distance to nearest predator, in-bush flag) | ✅ all present; needs M7 labels exported per step |
| Patch-revisitation choice GLM | **Yes — highest-value substantive import.** Food is discrete and depletes with a regeneration delay, so "patch" is well defined | `res_pos`, `res_active` (eat events are `res_active` 1→0 transitions with the agent on-cell), `agent_pos`, `animal_pos` for per-patch predator history | ✅ all present |
| Policy-entropy dynamics at eval time | No — computed then discarded | policy logits per step | ❌ recorder change |
| Ridge decoding of past/future position from `h_t` | Blocked | `h_t` (128-d) + `agent_pos` | ❌ recorder change |
| Allocentric vs egocentric decoding contrast | **Not applicable** — no heading state exists | — | n/a |
| Per-unit position-encoding GLM | Blocked, and see §5.4.2 on binning | `h_t` + `agent_pos` | ❌ recorder change |
| Decoder-weight modularity (past vs future overlap) | Downstream of decoding | ridge weights | ❌ |
| Value-function trace as a behavioural covariate | Blocked | `_value` per step | ❌ recorder change |
| Pruning $\times$ interpretability study | A new study, not an analysis (JaxPruner not in the stack) | — | new work |
| Auxiliary path-integration head | New architecture work; note this project deliberately has **no location sensor** | — | new work |

**The one enabling change.** Extending the eval recording payload with `h_t`,
`value`, and policy `entropy` / `log_prob` unlocks rows 7, 9, 11, 12 and 13 in one
move. Storage is not a barrier **[reviewer arithmetic]**: 128 float32 units
$\times$ 500 steps $\approx$ 256 KB per episode uncompressed, so a 200-episode
recording adds $\approx$ 51 MB before gzip. This is a code change and therefore **not
this reviewer's to plan or write — hand off to `senior-developer` for an
`issue_plan`**, noting that the recording-format version constant and the renderer
lockstep comment in `src/utils/eval_recording.py` both bear on it.

#### 5.4.2 Where the two environments differ enough to break a method

1. **Spatial scale: 9,216 cells vs 100.** This is the deepest incompatibility.
   ForageWorld's arena is 92$\times$ larger, and almost every spatial statistic in the
   paper presumes an arena the agent cannot cover. On a 10$\times$10 grid over a
   500-step episode, occupancy entropy saturates near its $\log 100 \approx 4.6$-nat
   ceiling for almost any non-degenerate policy; "coverage fraction" is uninformative;
   the exploration$\to$revisitation *phase transition* has no room to occur; and the
   outward-spiral search motif cannot be expressed. **Verdict: occupancy entropy,
   coverage, spiral-motif comparison and the phase-transition analysis do not transfer
   as evidence** — they can be computed but will not discriminate.
2. **The 14$\times$14 position binning has no analogue.** The paper coarse-grains
   9,216 cells into 196 bins to keep the per-unit GLM identifiable. Here there are
   100 cells and 128 hidden units, so a per-cell design matrix is already
   $100 \approx$ the unit count and there is no coarse-graining headroom. A per-unit
   position GLM would need heavy regularisation, and the radial-ramp analysis
   (which needs many radial shells) has at most $\sim$7 distinct radii on a
   10$\times$10 grid.
3. **Episode length: thousands of steps (decoding used $t \in [1000,6000]$) vs 500.**
   The paper's within-episode 75/25 decoder split gives thousands of training samples
   per episode; here it gives 375 train / 125 test for a 128$\to$2 map. Feasible, but
   the split design must change to **pooled-across-episodes with a by-episode
   held-out set**, which is a *different* control — it tests generalisation across
   layouts rather than across time within a layout.
4. **Random start position.** ForageWorld's entire allocentric frame is "displacement
   from the arena centre, which is the same cell every episode". This project sets
   `random_start_pos: true`, so no fixed world-anchored frame exists across episodes.
   Two consequences: (a) the target must be redefined as displacement from *this*
   episode's start cell, which converts the analysis from "world-anchored map" to
   "path integration"; (b) the paper's explanation for why a single decoder
   generalises across episodes — invariant global layout and orientation — **does not
   hold here**, since entity counts, positions and the start cell all vary per
   episode. A decoder that generalises across episodes here would be a *stronger*
   result than theirs.
5. **Vision is contact-only.** ForageWorld gives a 9$\times$11 tile window; this
   project runs `visual_sensor_range: 0` — the agent sees only what it is standing on
   — with an olfactory gradient as its only distal spatial sense. This **strengthens**
   the memory-demand argument (nothing is visible at a distance) but **breaks any
   method that assumes the agent perceives a scene**. Concretely, the revisitation
   GLM's `CowCount` regressor ("cows observed in that patch") has no analogue: this
   agent cannot count food at range. The nearest substitute is peak olfactory
   intensity experienced near that patch, which is a different construct and must be
   labelled as such.
6. **No heading / orientation state.** Movement is 4-connected with no facing
   variable, so the paper's egocentric decoder — its key specificity control — has no
   counterpart. Any port must find a different negative control (shuffled-$h$, or a
   task-irrelevant target such as the episode seed).
7. **Interoception is this project's subject and ForageWorld's furniture.**
   ForageWorld exposes health/food/drink/energy directly and noiselessly. This project
   deliberately **hides** injury and nutrition from the observation, delays pain
   perception through an alpha kernel ($\tau = 3$, length 12), and injects
   state-dependent perceptual noise per modality. That difference creates a decoding
   target the paper never considers and that fits this project far better than spatial
   decoding: **decode the hidden physiological state (`injury_level`, `nutrition`)
   from $h_t$** — does the recurrent state infer its own unobserved body state, and
   over what lag? That is the interoceptive analogue of the paper's position decoding,
   it uses exactly the same ridge machinery and the same $\Delta t$ sweep, and the
   ground truth is **already in the recordings** (`snapshots[t]['injury_level']`,
   `['nutrition']`) even though the agent never observes it. This is, in this
   reviewer's judgement, the single most valuable idea to lift from this paper.
8. **A behaviour class ForageWorld lacks.** This project has bushes with
   `hides_agent: true` and a pre-registered bush-dive measure (M2). ForageWorld offers
   only fight or flee. The project's defensive repertoire is therefore *richer* than
   the paper's, and the PPO-vs-PQN "fight vs flee" strategy-divergence result maps
   onto a three-way rather than two-way comparison here.
9. **Performance axis.** ForageWorld plots return; this project plots survival steps.
   Because ForageWorld's reward is a bounded homeostatic maintenance signal, return
   $\approx$ length, so the two axes nearly coincide *there* — but any ported figure
   must use survival steps here, per project rule.

#### 5.4.3 Claims this project is positioned to test or challenge

**(a) The headline claim, stripped of its confound.** The paper asserts that
model-free RNN agents show planning-like structure "purely through emergent dynamics",
while its own ablation shows the spatial code requires a supervised auxiliary
objective. This project trains **no auxiliary spatial objective and provides no
location sensor** (`location_sensor: false`), so it is an unusually clean test bed for
the deflated claim: *does a purely PPO-trained GRU, with contact-only vision, carry a
linearly decodable trace of its own past and future trajectory?* A positive result
would support the emergence claim **beyond** what the paper itself establishes; a
negative result would confirm that the auxiliary loss, not the RL objective, does the
work. Both outcomes are informative, and the experiment needs no new environment — only
the recorder extension in §5.4.1 and a ridge fit.

**(b) The 50–100-step "planning horizon" versus the discount horizon.** As derived in
§5.3.3, ForageWorld's $\gamma = 0.99$ gives $1/(1-\gamma) = 100$ steps, numerically
equal to the reported horizon. This project runs $\gamma = 0.95$, i.e. a 20-step value
horizon, with otherwise the same algorithm family and recurrent core. **A
decoding-horizon-versus-$\gamma$ sweep is a direct, cheap and decisive test of the
paper's most quotable number**: if the decodable horizon tracks $1/(1-\gamma)$, the
"planning horizon" is a restatement of the discount factor; if it is invariant, the
planning reading survives. This project is unusually well placed to run it because
$\gamma$ is a single agent-config key and the architecture is otherwise fixed.
**Experiment design hand-off: `experiment-designer`.**

**(c) Future-decodability as evidence of planning — the missing surrogate control.**
The paper never rules out that future position is decodable simply because motion is
smooth. The control is to decode $Y_{t+\Delta t}$ from *current position and recent
velocity alone*, with no hidden state, and require $h_t$ to beat it. **That control
arm requires no new logging whatsoever** — `agent_pos` is already recorded — so this
project can build and validate the surrogate baseline *before* the recorder change
lands, and then apply it the moment $h_t$ becomes available. This is the cheapest
available way to hold the paper's central inference to account.

**(d) Location-history effects on choice, versus in-the-moment reactions.** Prior work
in this repository (the chasing-rabbit study behind the `/trajectory-story` skill)
established two lessons that bear directly on this paper: aggregate statistics hide
conditional behaviour, and the avoidance observed there was **post-contact rather than
pre-emptive**. The paper's revisitation GLM — with a per-patch predator-encounter
history regressor — is exactly the instrument for the untested next question: do this
project's agents avoid *places* with a bad history, as opposed to avoiding predators
*in the moment*? That is a memory claim, not a reactivity claim, and it is currently
unmeasured here. All required inputs are already in the recordings.

**(e) Sparsity and interpretability.** "90% pruning leaves performance intact and
improves decoding" is untested in this stack and would be a new study rather than a
re-analysis. Flagging it as a candidate, not a recommendation.

**Hand-off summary (per this reviewer's scope — no code, config or script changes are
proposed as instructions):**

- Eval-recording extension to persist `h_t`, `value`, and policy `entropy` /
  `log_prob` (all three already computed and discarded in the rollout scan) →
  **`senior-developer`** for an `issue_plan`.
- Decoding-horizon-versus-$\gamma$ experiment, and any auxiliary-objective variant
  → **`experiment-designer`**.
- If any equation in §5.3 is lifted into a submission, verify the transcription
  against the OpenReview camera-ready → **`math-reviewer`**.

---

### 5.5 Appendix: Section-by-Section Backbone

The paper walked through in its own order. Everything in §5.2–§5.4 is a
reorganisation of the material below.

**Abstract.** Understanding DRL agents needs more than reward-curve comparison;
behavioural-analysis methods are underdeveloped in DRL. The authors apply neuroscience
and ethology tools to DRL agents in **ForageWorld** — a novel, complex, partially
observable environment with sparse depleting resource patches, predator threats and
spatially extended arenas. Contrary to common assumptions, model-free RNN-based DRL
agents exhibit structured, planning-like behaviour purely through emergent dynamics,
without explicit memory modules or world models. The tools are distilled into a
general analysis framework linking behavioural and representational features to
diagnostic methods.

**§1 Introduction.** Open-ended RL tasks involve partial observability, memory,
planning and spatial reasoning, yet there is no systematic understanding of the
strategies agents use, and no standard method for analysing or comparing behaviour
across tasks. Evaluation focuses on reward curves and aggregate performance. The
proposal is a behaviour-first approach modelled on animal neuroscience and ethology,
grounded in ForageWorld (built on Craftax, extended with patchy resources and
structured threats). Four contributions: the environment; adapted tools (path
analysis, GLMs, recurrent-state decoding); the finding that model-free RNN agents show
planning-like behaviour; a released pipeline and codebase. Five motif classes are
analysed: exploration, decision-making, learning dynamics, internal representations,
generalisation.

**Table 1 — the analysis framework.** Six rows: Goal Inference (decision GLMs,
in-context learning study, behaviour phase segmentation); Memory Span (RNN state
decoding, memory-module ablations); Planning Horizon (future decoding, auxiliary-loss
analysis); Spatial Structure (occupancy entropy, revisitation tracking, path
analysis); Network Capacity (size/pruning sweeps); Representations (decoding, GLMs,
encoding profiles).

**§2 Background.** §2.1 — generalisation in RL is a bottleneck; open-ended tasks scale
with the learner and carry naturalistic structure. §2.2 — open-ended benchmarks have
not attracted sustained neuroscience interest; existing brain-inspired work on ML
systems is mostly representational-similarity comparison, and similarity metrics are
known to yield inconsistent conclusions. Neural activity alone is insufficient to
explain computation; hence behaviour-first.

**§3 Methods.** Code at `github.com/RileySE/Craftax-Foraging/tree/foraging`.

- **§3.1 ForageWorld.** Cows have an arena-wide count limit, spawn at fixed locations
  and diffuse, producing temporary localised depletion that requires leaving and
  revisiting patches. Lakes maintain hydration. Predators appear near food patches and
  pursue on line of sight. Agents manage hunger, thirst, fatigue and health; fatigue
  recovers only through sleep (gated on energy below 50% of max, immobilising);
  starvation and extreme thirst reduce health; death at health zero. Arenas
  procedurally generated with randomised layouts. **Eq. 1** is the survival-focused
  reward:

$$
R(s_t) = 0.1\big[1 + \operatorname{sign}(\text{health}_t-5) + \operatorname{sign}(\text{food}_t-5) + \operatorname{sign}(\text{drink}_t-5) + \operatorname{sign}(\text{energy}_t-5)\big]
$$

  — reward for each resource kept above half max, correlating strongly with survival time without promoting overconsumption
  (design justification cited to EVAAA, ref. [66]).
- **§3.2 Architectures.** PPO with a GRU core (PPO-RNN). Optional auxiliary path-
  integration head predicting $(x_t,y_t)$ relative to the first-timestep origin from
  the shared $h_t$, trained jointly with **Eq. 2**,
  $\mathcal{L}_{\mathrm{aux}} = \mathbb{E}_t[\lVert\hat p_t - p_t\rVert_2^2]$, weighted
  by $W_{\mathrm{aux}}$. Biologically-inspired sparsity via JaxPruner magnitude
  pruning to 90%, motivated by 70–94% biological sparsity. Pruned agents perform
  comparably and sometimes decode better.
- **§3.3 Decoding allocentric position and planning horizons.** Record $h_t$ and
  $(x_t,y_t)$; at each $t$ train a decoder predicting displacement
  $Y_{t+\Delta t} = (\Delta x, \Delta y)$ for $\Delta t$ steps past or future from
  $h_t \in \mathbb{R}^{512}$ alone. Separate allocentric and egocentric decoders. Ridge
  regression for interpretability. Trained on the first 75% of each episode, evaluated
  on the final 25%. RMSE against an average-displacement baseline. All behavioural
  analyses in held-out test arenas with frozen weights.

**§4 Results.**

- **§4.1 Structured exploration and revisitation.** Trajectory visualisation,
  behavioural segmentation and spatial entropy show phase-like foraging analogous to
  mice in a labyrinth. Standard metrics confirm competence (long survival, directly
  proportional to return). Agents generate outwardly spiralling, azimuthally rotating
  loops from the start position that expand to cover most of the arena; exploration
  persists during revisitation; extended trajectories support predator avoidance and
  shortcut discovery. Early exploration qualitatively resembles insect search.
- **§4.2 Multi-objective foraging and strategic patch use.** Rapid transition from
  broad exploration to targeted revisitation; periodic returns to the start position
  (possible reorientation strategy). Agents are given no explicit patch list;
  revisitation emerges from recurrent dynamics. Substantial performance gap between
  PPO with and without recurrence. Expert action rates: eat 0.011 per step (stated as
  once per $\sim$95 steps), drink 0.034, sleep 0.262. Revisitation choices integrate
  spatial and task-relevant features multi-objectively (Fig. 3).
- **Figure 2 (ablations motivating deeper analysis).** Feedforward replacement and
  64-unit downsizing both impair badly; a 128-unit network underperforms when pruned;
  pruning at full size does not degrade training but improves spatial
  interpretability; removing the auxiliary loss reduces performance in large arenas
  only; a forward-facing FOV improves early learning while final performance hides
  behavioural differences. Curves are time-weighted EMA means across 5 seeds, shaded
  $\pm1$ s.d.
- **Figure 3 (revisitation GLM).** Coefficients for patch-history variables predicting
  the choice to revisit one patch over others, with choices defined 50 timesteps before
  each patch-eat event: prefer fewer prior eats (EatRate); no water-proximity effect
  (DrinkRate); avoid patches with more predator encounters (PredRate); prefer more
  recently visited (Recency); mild preference for longer dwell time; prefer more
  observed cows (CowCount); prefer higher prior position-prediction error
  (Uncertainty). Significance $*<0.05$, $**<0.01$, $***<0.001$; 95% CI error bars.
- **§4.3 Emergence of behavioural competencies over training.** Staged learning: from
  undirected exploration to structured goal-aligned strategies, only partly visible in
  reward curves. Early "fishing" (stationary near origin); after $\approx$20,000
  training iterations an abrupt multi-metric shift — longer travel in both early and
  late phases, better water/food trade-offs, sharp increase in tool-making, gradual
  gains in predator defence. Front-FOV agents explore farther earlier and search more
  densely. PQN-LSTM performs comparably to PPO-GRU with similar learning histories and
  pathing but much lower predator-killing rates: PPO converges to fighting, PQN to
  evasion.
- **Figure 4 metric panel.** Spatial uncertainty (normalised by distance from origin);
  distance from origin early vs late (before/after the first 1500 timesteps);
  state-occupancy entropy of position; angular orientation variance over 250-timestep
  intervals; predator FOV exposure; tool-making rate (1 = one tool, 2 = both);
  food/water satiation. 95% CI error bars.
- **§4.4 Recurrent state representations support memory and planning.** Position
  relative to origin decodes from $h_t$ up to 50–100 steps into past and future.
  $\approx$100/512 units are position-sensitive by per-neuron GLM; the response to
  position change increases with radial distance from origin, suggesting a
  distance-accumulation circuit modulating origin-revisitation. Decoding improves over
  training. Egocentric orientation cannot be reliably decoded.
- **Figure 5.** (A) one decoding model per agent, allocentric vs egocentric; (B) late-
  training egocentric decoding at chance across models; (C) allocentric decodable
  above chance to $\approx$50–100 steps past and future, with training-arena count
  varied to rule out arena-specific cues; (D) allocentric decoding improves over
  training, with larger error bars because decoding was limited to timesteps
  1000–6000 per arena. Chance baseline = average displacement per timestep; 95% CIs.
- **§4.5 Generalisation and modularity.** Pruned networks decode moderately better
  (Fig. 16). Removing the auxiliary objective eliminates above-chance decoding
  (Fig. 17) — so the associated performance loss stems from inability to encode
  position under the PPO objective alone. Past and future encoding use overlapping but
  functionally modular subpopulations that diverge at longer horizons (Fig. 18). A
  single decoder generalises across episodes, attributed to the preserved 96$\times$96
  layout and global orientation providing a stable allocentric reference frame.

**§5 Discussion.** Two findings: ForageWorld induces planning/memory-relevant
behaviour–neural motifs, and the toolkit reveals them where simple performance metrics
cannot. Challenges the view that sophisticated planning and memory require explicit
world models, mammalian-like brains, or symbolic memory. Claims the first systematic
evidence that planning-like behaviour emerges in complex naturalistic environments
without world models. Architecture choices — recurrence, pruning, auxiliary losses —
influence both performance and interpretability; pruning improves decoding while
leaving behaviour unchanged; the auxiliary objective improves both predator-related
behaviour and spatial encoding structure, a "win-win" consonant with the observation
that biological agents do not simply maximise reward. Relevance to small nervous
systems: comparable behaviour emerges with only a few hundred recurrent units.
Broader argument: DRL models are not too complex for neuroscience-style analysis — if
neuroscience tools cannot explain DRL agents given full behavioural and neural access,
their utility for brains is limited; simulation also addresses low statistical power,
data cost and limited experimental control in naturalistic neuroscience.

**Limitations & Broader Impacts.** (i) Craftax's 96$\times$96 grid cap prevents
testing generalisation to larger arenas. (ii) Precise locations of all trees, rocks
and lakes could not be logged, limiting landmark-navigation analysis. (iii)
Instrumenting a wider range of RL architectures with full behavioural–neural logging
is currently non-trivial, hence the focus on PPO and PQN with shared components.
Positive societal impact anticipated via closer collaboration between neuroscience,
cognitive science, ethology and RL.

**§6 Acknowledgments.** NIH, James S. McDonnell Foundation, Simons Foundation,
McKnight, CIFAR, NSF, Harvard Medical School awards; Kempner Institute established by
a Chan Zuckerberg Initiative Foundation gift; Fulbright/Jansons Legat/Norwegian State
Educational Loan Fund for F.B.B.

**References.** 118 entries spanning open-ended RL benchmarks, RL generalisation,
neuroscience-of-behaviour position papers, foraging neuroscience, insect navigation,
hippocampal replay and cognitive maps, sparsity/lottery-ticket work, and the tooling
cited for analysis (`bayesmove`, `ctree`, NeMoS). Ref. [66] is EVAAA (Lee et al.,
NeurIPS 2025 D&B), cited for the homeostatic reward design; ref. [15] is Craftax;
ref. [79] is PQN; ref. [67] is JaxPruner.

**Appendix A — ForageWorld task and training details.** Design balances task
complexity, transparency to analysis, training efficiency (JAX GPU acceleration),
biological plausibility and connection to recurrent-memory architectures of interest
to neuroscientists. Explicit contrast with Minecraft (hard to instrument, compute-
prohibitive for statistical analysis) and with n-armed bandits (no spatial or temporal
component). Agents need billions of timesteps; Craftax remains unsolved by any agent.

- **A.1 Procedural arena generation.** Fixed 96$\times$96 grid; cow and lake spawn
  points initialised in random patches using Perlin noise; obstacles break line of
  sight; agent begins at arena centre.
- **A.2 Episode structure.** Max 100,000 timesteps or health zero — the typical
  outcome even for competent agents.
- **A.3 Sensory input.** 9$\times$11 grid-centred egocentric view plus an inventory
  vector containing health, food and water satiation, fatigue and collected items;
  encoded by a feedforward layer before the recurrent network.
- **A.4 Logging infrastructure.** Per timestep: full recurrent hidden state, internal
  reward components, movement trajectories, environmental state, agent decisions —
  complete schema in Table 2.
- **A.5 Training hyperparameters.** PPO with GAE; 3 billion timesteps per agent
  (hundreds of thousands of episodes); rollout horizon 64; minibatch 8192; learning
  rate 0.00025 (Table 3 lists 0.0002); Adam with default momenta; gradient-norm clip
  1.0. One A100 or H100 per run, 24–48 h, $\ge$24 GB typical. Logging every 2048 PPO
  iterations ($\approx$134 M timesteps); best checkpoints selected on held-out arena
  performance.
- **A.6 PPO-RNN architecture.** **Eq. 3** is the clipped surrogate with
  $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$; full loss adds
  value and entropy terms. Single-layer GRU, 512 hidden units, shared by actor and
  critic, both heads reading $h_t$; GRU preferred over LSTM for efficiency and
  interpretability; implementation follows Craftax. $\approx$6.5 M trainable
  parameters.
- **A.7 Future directions.** Curriculum learning (staged competency emergence suggests
  ForageWorld suits it; Craftax is compatible with JAX curriculum packages); testing
  LLM foraging behaviour (navigation is under-explored for LLMs, but transformers are
  harder to analyse than RNNs and Craftax lacks language labels for state variables);
  multi-agent support exists in recent Craftax but not yet in ForageWorld.

**Appendix B — interpreting agent behaviour.**

- **B.1 Exploration and revisitation trajectories.** Three episodes, each shown as
  early exploratory loops, a transition-phase path, and the full trajectory with
  revisited patches highlighted (Fig. 6).
- **B.2 Behavioural phase segmentation via unsupervised clustering.** `bayesmove`
  (nonparametric Bayesian, Cullen et al.) over turning angle and step size on a
  7-timestep moving window yields three latent movement states; conditional inference
  trees (`ctree`) identify which task variables predict transitions. States map to
  short- (3), mid- (1) and long-range (2) navigation; state 1 coincides with predator
  events, state 3 with food and eat actions. Key `ctree` predictors: food level,
  positional uncertainty, `enemy_present`, `num_passives_nearby`. Run on test arenas
  with fixed weights, so no weight updates are involved (Fig. 7).
- **B.3 Comparison to real insect search patterns.** Ant search after nest
  displacement (adapted from Wehner & Srinivasan) and bee first/second/third departures
  (adapted from Osborne et al.) both show outwardly expanding, azimuthally rotating
  loops matching the agents' early exploration (Fig. 8).
- **B.4 GLM outputs and multicollinearity checks.** Logistic GLMs classify whether a
  patch was chosen at a revisitation point from historical variables; one model fit
  jointly across five PPO-RNN agents over 7,978 revisitation decisions, agent ID as
  fixed effect, `statsmodels.formula.api`. Uncertainty, predator-encounter rate and
  recency are significant; dwell time and drink rate contribute less. All VIF $<10$
  (Fig. 13).
- **B.5 Training dynamics and entropy evolution.** Average policy entropy declines and
  log-probability of the selected action rises, reflecting increasing confidence; late
  in training entropy rises slightly as the agent finds solutions that satisfy both
  performance and the entropy bonus (Fig. 9).
- **B.6 Memory and predator-related ablations.** PPO without recurrence gets trapped in
  local loops with markedly reduced survival (Fig. 10). Ablating the auxiliary
  path-integration objective barely changes movement patterns but degrades predator-
  killing emergence (Fig. 11) and destroys decoding (Fig. 17). A predator-blind
  variant (agent cannot damage predators) shows similar movement and comparable
  survival, indicating evasion rather than combat is the primary strategy (Fig. 12).

**Table 2 — logged variables.** Action; Health; Food; Drink; Energy; Done; Is
Sleeping; Is Resting (disabled in all experiments); Player Position (absolute, origin
top-left of the 96$\times$96 arena); Recover / Hunger / Thirst / Fatigue timers; Light
Level (modifies predator spawn rate, higher at night); Distance to Melee (L1); Melee on
Screen; Distance to Passive (nearest cow); Passive on Screen; Distance to Ranged;
Ranged on Screen; Num Melee Nearby; Num Passives Nearby; Num Ranged Nearby; Delta X &
Y (relative to start, arena centre); Predicted Delta X & Y (auxiliary head output); Num
Monsters Killed; Has Sword; Has Pick; Held Iron; Value; Entropy; Log Probability;
Episode ID.

**Table 3 — hyperparameters.** Model: LR 0.0002; $\gamma$ 0.99; $\lambda_{\rm GAE}$
0.8; PPO clip $\epsilon$ 0.2; $W_V$ 0.5; $W_{\rm entropy}$ 0.01; activation $\tanh$;
512 neurons per layer; 64 steps per PPO iteration; 1024 parallel environments; 4 epochs
per iteration; 8 minibatches per epoch; 3$\times10^9$ total timesteps;
$W_{\rm aux}$ 0.025 (swept over $\{0.01, 0.1, 0.025, 1.0\}$); magnitude pruning (only
JaxPruner type that did not severely degrade performance); prune step 20,000
(mid-training). PQN: $\epsilon$ start 1.0, finish 0.05 (swept
$\{0.005,0.01,0.05,0.1\}$), decay 1.0 (swept $\{0.1,1.0,2.0\}$); 1.2$\times10^{10}$
total timesteps; 6$\times10^9$ total decay timesteps (swept $\{1,6,12\}\times10^9$);
128 env steps per update; $\lambda$ 0.5. Environment: predators True; max cows 108
(swept $\{48,72,108\}$ to balance patch-leaving and revisitation); full action space
False; map size 96 (chosen from $\{24,48,96\}$ to maximise navigation and memory
challenge); directional vision False.

**Appendix C — interpreting agent representations.**

- **C.1 Impact of auxiliary objectives on decoding.** Removing the auxiliary path-
  integration objective eliminates above-chance allocentric decoding — so the loss is
  not merely a regulariser but is essential for inducing spatially interpretable
  representations (Fig. 17).
- **C.2 Coefficient structure and functional modularity.** Ridge weights for past and
  future decoders aligned by neuron ID, for sparse and dense agents at 20- and
  50-timestep horizons; overlapping populations in both, with overlap diminishing at
  longer offsets and sparse networks showing greater separation and sparser weight maps
  (Fig. 18).
- **C.3 Regularisation and train–test split.** Some RNN units are inactive for entire
  episodes, making decoding ill-conditioned; hence ridge with

$$
\mathcal{L}_{\Delta t}(f) = \sum_{i=1}^{N}\big(Y^i_{t+\Delta t} - f(h^i_t)\big)^2 + \alpha\lVert f\rVert^2_K , \qquad f(h_t) = Ah_t + b, \quad A\in\mathbb{R}^{2\times512},\; b\in\mathbb{R}^2 .
$$


  Ridge chosen for interpretability, numerical stability and
  efficiency, with $O(Np^2)$ time complexity and direct access to coefficients. Split:
  first 75% of timesteps per episode for training, last 25% for evaluation, to prevent
  leakage from temporal continuity in $h_t$; decoders trained across multiple episodes.

**Appendix D — per-neuron position-encoding GLM.** NeMoS used to fit one GLM per
neuron predicting neural activity from current position; 5–10 episodes minimum per fit
to avoid over-fitting one arena configuration. The 96$\times$96 arena coarse-grained
into 14$\times$14 position bins with a binary regressor per bin (196 coefficients);
neural data shifted to the positive axis and normalised to $[0,1]$ per neuron. First
70% of each episode trained, last 30% held out (Fig. 19); most neurons have comparable
train and test performance (Fig. 20). Roughly 100/512 neurons fit well on position
alone (typically 60–120 per run; example fit Fig. 21). Coefficients clustered by
$k$-means, plotting the three largest-mean clusters out of five (Fig. 22), plus average
coefficient per position bin (Fig. 23). The number of strongly position-encoding
neurons and the average coefficient magnitude both increase with radial distance from
the origin (Fig. 24), suggesting an accumulation circuit that ramps with distance and
pressures return to origin — partially explaining periodic origin revisiting. The
pattern holds without the auxiliary objective (Figs. 25–26) and in PQN-RNN
(Figs. 27–28), "suggesting it is fundamental to the task solution". Example scripts
are in the paper's GitHub repository.

**Appendix E — alternative DRL objective (PQN).** PQN adapted to ForageWorld with the
LSTM architecture and default hyperparameters from Gallici et al.; same environment and
training hyperparameters as PPO except where Table 3 notes otherwise. Successful
PQN-LSTM runs match PPO-GRU performance (Fig. 29), but across the two successful
exploration configurations **only 30% of PQN runs performed comparably**, the remainder
failing catastrophically in a way never seen in PPO — attributed to suboptimal
exploration (early convergence to simple, reliable but suboptimal behaviour).
Successful runs show similar learning histories and pathing (Figs. 30, 31) but much
lower predator-killing rates, hence the fight-versus-flee strategy divergence.

**Appendix F — toward architectures for cognitive-map formation (future directions).**
Agents do not yet use structured maps; biological systems use both allocentric and
egocentric representations, particularly grid-based computations. Cognitive maps
support retrospection, prospective planning and flexible decision-making, and in
animals are modulated by goals, landmarks and reward events; grid-based
representations have been integrated into transformers and actor–critic agents.
Proposed next step: examine how grid-based recurrent architectures solve ForageWorld,
and apply the behavioural–neural framework to other architectures and tasks.

**Appendix G — Craftax licensing.** ForageWorld is modified from Craftax; the MIT-style
Craftax licence (Copyright © 2024 Michael Matthews) is reproduced in full.

**NeurIPS paper checklist.** Claims: Yes. Limitations: Yes (§5). Theory: NA (no
theoretical results). Reproducibility: Yes (code repository). Experimental settings:
Yes (Table 3 + repository). Statistical significance: Yes (95% CI unless otherwise
specified). Compute: Yes (appendix). Code of ethics: Yes. Broader impacts: Yes (§5).
Safeguards: NA. Licences: Yes (Craftax licence in Appendix G; figure reuse licensed —
one figure paid for, one under CC-BY). New assets: Yes (environment, model and
analysis code released). Human subjects / IRB: NA. LLM usage: NA (no LLMs used).
