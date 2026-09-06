---
title: "What the neuromodulator reads x what it re-tunes — a 16-cell single-seed screening grid"
topic: nmn_input_site_grid
status: active
created: 2026-09-07
last_updated: 2026-09-07
wandb_group: nmn_input_site_grid
wandb_tag: "rppo_nmnsite_t{1,2,3,4,5,16}<slug>_{ALL,I,X}_s42"
develop_link: docs/develop/active/neuromodulation/MODULATION_SITE_REFACTOR.md
---

# What the neuromodulator reads x what it re-tunes — a 16-cell screening grid

> **Status**: PRE-REGISTERED — **configs now written and verified; ready for pre-flight review.**
> The two config keys this design waited on are in the code (the modulation-site refactor's
> Part A and Part B, both landed 2026-09-07), and the one open question about how returns are
> computed has been answered in this design's favour, so nothing in §2.7 blocks a launch any
> more. All sixteen agent configuration files exist under
> `configs/models/recurrent_ppo/nmn_input_site_grid/`, were written by a committed generator
> rather than by hand, and every one of them has been loaded through the real training path and
> used to build an actual model — see §3.1.
> **Date**: 2026-09-07
> **Author**: `experiment-designer`
> **Mode**: fully pre-registered. Every prediction, threshold and failure-mode ruling in §2.5,
> §4 and §5 is fixed **before** any run launches and may not be adjusted afterwards.
> **Amended**: 2026-09-07, in answer to `plan-reviewer`'s **NOT READY** verdict. The headline
> behavioural measure was replaced (§4.2), the resolution bands were raised (§2.3) and eight further
> findings were answered — §10 records what changed, and the two places where I did something other
> than what the review asked for, with the reasoning. Every amendment is still pre-registration: all of it was written **before**
> any run in this grid launched and before any of its data existed.
> **Amended again**: 2026-09-07, second amendment — still before any run in this grid launched.
> Building the headline behavioural measure and running it against the five finished unmodulated
> baselines found a flaw in this pre-registration: the injury bins it registered describe a contrast
> the environment **cannot produce**. Those bins are re-registered (§4.2), every behavioural
> threshold is restated against the **five-seed band** rather than the within-run error bar (§2.5),
> the measurement's own result about the control agent is recorded as a finding (§4.2), and the
> post-training evaluation population now has committed spec files instead of a scratch script
> (§3.3). §C records the full list.
> **Open for the user**: §11 — a smaller, replicated alternative to this grid, recommended by the
> reviewer, recorded for the user to rule on. This design proceeds as approved in the meantime.
> **Related**: [[MODULATION_SITE_REFACTOR]] (the implementation this design consumes) ·
> [[return_mode_cmp_10M]] (source of the noise floor and of the return-mode choice) ·
> [[INJURY_HIDING_SIGN_RECONCILIATION]] (source of the behavioural-measurement rules) ·
> [[TRAJECTORY_STORE_SCHEMA]] (the store the behavioural measures are computed from) ·
> [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]] (an unresolved pre-launch dependency) ·
> [[project_plan]]

---

## 1. Research question

### 1.1 In plain language

The agent in this project carries a small second network — the **neuromodulator** — that reads the
agent's senses and continuously re-tunes the main policy network, the way a brain chemical such as
acetylcholine re-tunes cortex. Two things about that arrangement have never been varied
independently, because until now the code did not allow it:

- **What the modulator reads.** Today it reads the agent's entire sensory vector. Nobody has asked
  whether it needs all of it, or whether the *internal-body* channels alone would do — or whether
  the *external-world* channels alone would do just as well, which would undercut the whole story.
- **Which parts of the main network it re-tunes.** Today it re-tunes the sensory front-end and the
  memory cell, by two different mathematical operations, and cannot reach the decision-making
  layers at all. "Where it acts" has therefore always been tangled up with "how it acts".

This experiment crosses those two factors for the first time. A companion refactor makes every
re-tuning site use the **same** operation (multiply each neuron by a learned gain and add a learned
offset — the technique called **FiLM**) and makes each site switchable from the configuration file,
so that "where" can finally be varied on its own.

**The behaviour this is ultimately about**, in the user's own framing:

> The agent should show **context-dependent policy** — e.g. more bush-hiding when injury is high,
> and ignoring the same bush when injury is low.

The modulator is this project's proposed *mechanism* for that behaviour. If it works, the
agent's response to an external cue (a predator nearby, a bush nearby) should change with its
internal state. This grid asks which reading-and-writing arrangement, if any, produces that.

**A result measured before this grid launched, which changes what the grid is asking.** The headline
behavioural measure described below was built and run against the five already-finished *plain*
agents — no modulator at all — that this grid uses as its control. In those five agents **the target
behaviour is absent, and if anything leans the other way.** Starting an episode badly wounded
changes the chance that a step out of the open lands in cover by somewhere between **−0.53 and
+0.10 percentage points**, against a base rate of about **7 such steps in every 100**, and three of
the five agents move in the *negative* direction. What the wounded agent does instead is **stop
moving**: its resting rate rises by **18 to 21 percentage points**, an effect roughly **thirty times
larger** than anything the cover-seeking measure moves. So this grid is not asking whether the
modulator *amplifies* an existing tendency to seek cover when hurt. It is asking whether the
modulator **creates** one. The numbers, the method and the caveats are in §4.2.

### 1.2 What is being varied

**Factor 1 — what the modulator reads.** Three settings, named by short codes used throughout this
document. The environment is the 10x10 jump-attack world, whose observation is 27 numbers spread
across seven sensors.

- **ALL** — everything the agent senses (all 27 numbers).
- **I** — the body's own signals only: how full the agent is, and a smoothed internal ache that is
  the only trace it gets of its own wound (2 numbers). The agent cannot see its injury or its
  nutrition directly; these two channels are its entire interoceptive world.
- **X** — the outside world only: contact pain, smell, collision, vision (19 numbers).

**Factor 2 — which parts of the main network the modulator re-tunes.** Six settings, numbered as
the user chose them: **1** = nothing at all (the plain, unmodulated agent — this project's true
control), **2** = the sensory front-end, **3** = the memory cell, **4** = the action-choosing
layers, **5** = the value-estimating layers, **16** = all four of those at once.

The unmodulated agent has no modulator, so "what it reads" does not apply to it. The grid is
therefore **1 control + (5 modulated targets x 3 input slices) = 16 training runs**, one seed each.

### 1.3 What this experiment is, and is not

**It is a screen, not a confirmation.** Every cell gets one random seed. A recently completed
25-run study in this same world measured how much two runs of the *same* configuration differ
purely because of the random seed: for the exact settings this grid uses, five seeds of the
unmodulated agent finished between 162.4 and 166.9 survival steps — a spread of about 4.5 steps —
while arms of that study that landed in a fragile learning regime spread by 23, 61 and 101 steps.
**Any difference this grid finds that is smaller than that spread is not a finding.** §2.3 states
exactly what magnitude this design can and cannot resolve, and §4.4 names which comparisons must be
re-run with more seeds before anyone believes them.

**One contrast is deliberately not clean, and that is a recorded decision.** Proprioception — the
agent's sense of its own six possible actions — is in neither restricted slice, because it is
classically neither interoceptive nor exteroceptive. It therefore appears only under **ALL**. The
consequence, stated here so no future reader has to discover it: **ALL and X differ by 8 numbers,
not 2 — both interoceptive channels plus all six proprioception channels — so an ALL-versus-X
difference cannot be attributed to interoception.** The clean interoceptive contrast in this
design is **I versus X**, and only that one.

**A question about the shape of this grid is open for the user, and is not settled here.** Sixteen
cells at one seed each answer sixteen questions with no replication; the same sixteen GPU slots
could instead answer five of them at three seeds each. `plan-reviewer` recommends the second.
The argument, and what each option buys, is in **§11**. **This document proceeds with the 16-cell
grid the user approved** — §11 is a recommendation for the user to rule on, not a change to the
design.

### 1.4 Formal hypotheses

Each is translated into English on first mention above or in place below.

> **H1** (*the write site matters at all*): survival differs across the six target conditions by
> more than the seed-noise floor of §2.3.
>
> **H2** (*the sensory front-end is the right place to act*): the front-end-only target (2) beats
> the unmodulated control (1), and is not beaten by the memory, action or value targets alone.
> This is the site Paper 1 has already committed to; H2 asks whether that commitment survives a
> controlled comparison.
>
> **H3** (*re-tuning the value estimator helps*): the value-only target (5) beats the unmodulated
> control (1). This is the one cell carrying a directional prediction from outside this project —
> see §2.5.
>
> **H4** (*more sites is better*): the all-four target (16) beats every single-site target.
> **No directional prediction is offered for H4** — see §2.5.
>
> **H5** (*the modulator must read the body*), in **two parts**, because the version the project's
> thesis needs is not the version that is easy to satisfy:
>
> - **H5-add** — *the load-bearing claim*. An agent whose modulator reads only the interoceptive
>   channels (**I**) shows **more internal-state-dependent behaviour than the unmodulated control**
>   (cell 1). This is the claim that matters, because the main policy network already receives all
>   27 sensory numbers — both body channels included — in **every** arm of this grid, the control
>   included. If a body-reading modulator adds no state-dependence on top of that, then the
>   modulator is not the mechanism, whatever the I-versus-X comparison says.
> - **H5-spec** — *the specificity check*. At the same write target, **I** shows more
>   internal-state-dependent behaviour than **X** (exteroceptive channels only).
>
> H5-spec on its own is close to true by construction: X's modulator cannot read satiation at all
> and reaches the wound only indirectly through exteroceptive contact pain. It is therefore reported
> as a specificity check and never as the confirmation. (`plan-reviewer` finding 7.)
>
> **H6** (*restricting the input costs survival*): survival falls monotonically with the number of
> input channels (ALL 27 > X 19 > I 2), at the same write target. **No directional prediction** —
> H6 is registered as an *alternative reading* to be checked against, so that a capacity effect is
> not mistaken for an interoception effect.

**H5 is the hypothesis this project's thesis rests on.** It is also the one whose primary evidence
is behavioural rather than survival-based, because a modulator can raise survival without producing
context-dependent policy at all (§4.2).

---

## 2. Experimental design

### 2.1 Independent variables

**Factor 1 — modulator input slice** (`agent.modulation.input_sensors`, delivered by Part B of the
refactor; see §2.7).

| Code | Sensor names passed to the modulator | Dims | Rationale |
|---|---|---|---|
| **ALL** | `"all"` | 27 | Status quo; every historical neuromodulated run in this project. |
| **I** | `Satiation`, `Interoceptive Nociception` | 2 | The agent's entire interoceptive channel set under this environment config. Injury and nutrition are not observable, so these two are all the body-state information that exists. |
| **X** | `Extero Nociception`, `Olfaction`, `Collision`, `Visual` | 19 | Every exteroceptive channel. |

Observation layout, verified against a live run banner and against
`src/environment/sensor.py::get_observation_breakdown`: Satiation 1, Interoceptive Nociception 1,
Extero Nociception 1, Olfaction 5, Collision 5, Proprioception 6, Visual 8 = **27**. The
environment sets `injury_observable: false` and `nutrition_observable: false`
(`configs/environment/default.yaml:256-257`), so the agent perceives its own wound only through the
alpha-kernel-smoothed `Interoceptive Nociception` channel (`injury_smoothing_duration: 3`).

**Proprioception (6 dims) is in neither restricted slice.** This is a user decision, recorded on
2026-09-07: proprioception is classically neither interoceptive nor exteroceptive, so it appears
only under ALL rather than being arbitrarily assigned to one side. Its consequence is confound
**C2** in §2.4 and is stated in §1.3.

**Factor 2 — modulation write target** (`agent.modulation.sites.*`, delivered by Part A).

| # | Cell slug | `type` | `sites.encoder` | `sites.rnn` | `sites.actor` | `sites.critic` | What it re-tunes |
|---|---|---|---|---|---|---|---|
| **1** | `t1none` | `null` | — | — | — | — | Nothing. The unmodulated control. No modulator is constructed. |
| **2** | `t2enc` | `FiLM` | true | false | false | false | The sensory front-end (both the per-sensor stage and the fusion hub — one atomic site). |
| **3** | `t3rnn` | `FiLM` | false | true | false | false | The memory cell's emitted output (never its carried state — refactor decision D5). |
| **4** | `t4act` | `FiLM` | false | false | true | false | The single hidden layer of the action head, pre-activation. |
| **5** | `t5crt` | `FiLM` | false | false | false | true | The single hidden layer of the value head, pre-activation. |
| **16** | `t16quad` | `FiLM` | true | true | true | true | All four at once. |

The grid is **1 + 5 x 3 = 16** cells, one run each.

### 2.2 Controlled variables

Everything below is held identical across all 16 runs and must be verified from each run's own
trainer-written `models/config.yaml`, not from a fresh reload of the source YAMLs.

**Environment** — `configs/environment/experiment/basic/04-jump_attack_10x10.yaml`, unmodified and
shared by all 16 runs. 10x10 grid, 500-step episode cap, 1-4 regenerating food items, 6-12 static
damaging rocks, 0-2 hunting predators with a 2-3 cell pounce at 50% hit rate, 0-2 harmless rabbits,
scattered concealing bushes. Random starting position, random starting nutrition, random starting
injury. Six actions: four moves, Rest, Eat. Homeostatic reward (`use_homeostatic_reward: true`):
each step's reward is the reduction in the L2 distance of (satiation, injury) from their setpoint,
so **injury is an argument of the reward and healing is rewarded directly** — the precondition the
literature attaches to value-head modulation (§2.5, H3).

**No new environment config is created.** Both independent variables are agent-side. Producing a
copy of `basic/04` under this topic's directory would create a second file that can silently drift
from the original and would break the direct comparability with the 25-run return-mode study that
this design leans on.

**Agent hyperparameters** — identical to `configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml`,
which is the unmodulated MC arm of the return-mode study, so cell 1 is a lineage-exact continuation
of an existing 5-seed record (subject to C7):

```yaml
algorithm: RecurrentPPO      rnn_type: GRU          hidden_size: 128
lr_actor: 0.0005             activation: relu       sequence_length: 128
gamma: 0.95                  K_epochs: 4            eps_clip: 0.1
entropy_coef: 0.01           gae_lambda: 0.95       vf_coef: 0.5
max_grad_norm: 0.5           use_layer_norm: true   frame_stack: 1
fc_layers: [128,128]         actor_fc_layers: [128,128]   critic_fc_layers: [128,128]
encoding_mode: hierarchical  hierarchical_params: default_mlp [128,128],
                             unimodal_overrides {visual, olfaction} [128,128],
                             multimodal_hub [128,128]
return_mode: MC
```

**Modulator hyperparameters** — identical across all 15 modulated runs:

```yaml
type: FiLM                 mod_hidden_size: 16      grouping_size: 1
percept_bias_init: 3.0     percept_add_bias_init: 0.0
memory_bias_init: 0.0      memory_clip: [-2.0, 2.0]
rnn_mechanism: activation  temperature: {enabled: false}
```

Four of those settings are load-bearing and are pinned for stated reasons:

- **`return_mode: MC`.** The 25-run study established that keeping the value-head's training target
  and the advantage on **one scale, near spread 1**, is worth roughly an order of magnitude in
  experience efficiency (measured range 6x-14x) and gives far tighter seed reproducibility (5-seed
  standard deviation 1.9 steps against 10-26-51 for the split-scale settings). MC is also the
  matched-scale mode every historical neuromodulated run in this project used, so keeping it
  preserves comparability with that record.
- **`grouping_size: 1`** — one gain and one offset per neuron, the finest available. User's choice;
  it is also the setting with the highest measured gain diversity in the prior grouping sweep, so
  if a site cannot express a context-dependent gain at `g1` it will not at a coarser grouping.
- **`rnn_mechanism: activation`** — the uniform FiLM operator, not the legacy gate-bias operator.
  This is what makes Factor 2 a clean single-factor variable: under refactor decision D6, any run
  that is not `gate_bias` uses the **same plain GRU cell as the unmodulated control**, which
  removes a known initialisation confound present in every historical modulated-vs-baseline
  comparison. The cost is stated as confound **C4**: these runs are therefore *not* continuous with
  the historical neuromodulated record on the memory site.
- **`temperature: {enabled: false}`** — the policy-temperature scalar is a fifth, differently-shaped
  modulation channel (it divides the action scores rather than scaling neurons). Leaving it on
  would put it in every modulated cell and confound all five targets with it. User's choice; it is
  also the refactor's new default.

One setting in that block is **inert** and is listed only so the config files can be read against
this design:

- **`percept_bias_init: 3.0` does nothing under `type: FiLM`.** The gain head's output bias is
  initialised to 1.0 for FiLM regardless of this key (`src/models/neuromodulator.py:142-146`); the
  value 3.0 is the legacy `Gate`-operator default that the config family carries forward. It is
  **not** load-bearing here, and no reading of these runs may attribute anything to it.
  (`plan-reviewer` finding 10.)

**Training budget** — 10,000,000 episodes per run, matching the return-mode study exactly, so cell 1
is directly comparable with its five existing seeds. **Seed 42** for all 16 runs, taken from the
config-owned value in `configs/train/default.yaml:83`; no `--seed` flag is passed.

**Not matched, and cannot be:** total parameter count. The modulator's input layer has 27, 2 or 19
input columns depending on the slice, so the three input arms differ slightly in parameter count.
See confound **C3**.

### 2.3 What this design can and cannot detect

**The noise floor, measured rather than assumed.** From the 25-run return-mode study, run on this
exact environment with this exact agent configuration at this exact budget:

| Arm of that study | 5 seeds, survival steps at end of budget | Spread (max − min) | Standard deviation |
|---|---|---|---|
| `GAE_NORM` (tightest) | 169.9, 170.0, 170.1, 170.6, 171.2 | **1.3** | 0.52 |
| **`MC`** — this grid's exact setting | 162.4, 164.2, 165.7, 166.8, 166.9 | **4.5** | **1.89** |
| `GAE` | 112.2, 120.3, 131.5, 134.7, 135.2 | 23.0 | 10.14 |
| `MC_FIXED` | 61.9, 81.8, 88.1, 118.7, 122.6 | 60.7 | 25.70 |
| `MC_RAW` | 35.5, 45.0, 45.3, 131.7, 136.7 | 101.2 | 50.73 |

Read at the largest budget all 25 runs reached (349 M environment steps) the same ordering holds
with wider spreads: 2.7 steps for the tightest arm, 9.4 for `MC`, and 26 to 83 for the three slow
arms.

**Two lessons, both pre-registered into the thresholds below.**

1. In the well-behaved regime — an arm that learns normally and plateaus — seed dispersion for this
   configuration is about **2 steps of standard deviation and under 10 steps of full range**.
2. In a fragile regime — an arm that is still climbing, or that has landed in a low-performance
   trap — dispersion explodes to **20-100 steps**, and an arm mean can describe no individual seed
   at all. A single-seed number from such an arm carries essentially no information about its
   condition.

**Pre-registered resolution bands** for the difference between any modulated cell and the
unmodulated control, on survival steps at matched environment experience. **These bands were raised
on 2026-09-07** after `plan-reviewer` (finding 5) caught an arithmetic error in their justification:
the original floor of 10 steps was described as "about 2x the largest observed 5-seed range", but
that range is 9.4 steps, so 10 is **1.06x** of it — the floor sat exactly at the noise edge with no
margin at all. The corrected bands are 15 and 30.

| Difference vs. control | Ruling | Justification |
|---|---|---|
| Below **15** steps in absolute value | **No evidence.** Not reported as an effect in any direction. | 1.6x the largest observed 5-seed range for this configuration (9.4 steps at matched budget), which leaves a real margin for the untested assumption that a modulated arm is as seed-tight as an unmodulated one (assumption A3 of the review). |
| **15 to 30** steps | **Flagged, unresolved.** Reported as a candidate only; may not be described as an effect. | Above the observed range with margin, but not by enough to survive one bad draw in a fragile regime, where the same study measured 20-100 step dispersion. |
| Above **30** steps | **Candidate real effect.** Promoted to a 5-seed confirmation before any claim is made. | 3.2x the observed range and 16x the `MC` standard deviation. Still a candidate, never a claim: n = 1. |
| Any arm judged fragile by the §5 criteria | **Unresolved regardless of margin.** | Lesson 2 above. |
| Any arm ruled **trap, unresolved** by §5 item 8 | **Unresolved. Not counted as a negative result and not eliminated.** | A site that draws one bad seed and settles into a low plateau is indistinguishable, at n = 1, from a site that genuinely does not work. See §5 item 8. |

**Elimination is not symmetric with promotion, and that asymmetry is stated rather than hidden.** A
positive result above 30 steps must be re-run at 5 seeds before it becomes a claim; a negative result
has no such gate, so a good site that draws a bad seed is quietly dropped. The prior study shows how
real that is: five seeds of the **same** `MC_RAW` configuration finished between 35.5 and 136.7
survival steps (the table above) — one draw from that arm would have described it as either dead or
competitive, depending entirely on the draw. The **trap, unresolved** rule of §5 item 8 is the mitigation — it is
deliberately conservative, and it means this grid will hand back fewer clean eliminations than a
16-cell screen appears to promise. (`plan-reviewer` finding 6; see also §11.)

**Nothing in this grid is confirmatory.** All 15 modulated cells are n = 1, so all 105 pairwise
contrasts among them are one draw against one draw. The grid's job is to **rank and to eliminate**,
producing a short list for a properly seeded follow-up — not to establish anything.

**One exception, and it is about precision, not generalisability.** The behavioural measures of
§4.2 are computed over **300,000 evaluation episodes per checkpoint**, so their *within-run*
confidence intervals are very narrow even at one seed. That narrowness measures how precisely each
agent's behaviour has been characterised; it says **nothing** about whether a second seed of the
same configuration would behave the same way, and the gap between those two things is large.

**That gap has now been measured, and it is a factor of about six.** On the five existing
unmodulated runs, the within-run 95% confidence interval on the headline behavioural statistic
`Δ_B0` is about **±0.10 percentage points**, while the five-seed band — the range those same five
identically-configured runs span for no reason but the random seed — is **0.63 points wide**
(§4.2). The consequence is stated here because it is the single easiest mistake this design could
invite: **an arm reading `Δ_B0 = +0.3` points would clear its own error bar comfortably and still
sit inside the null band.** Both numbers are reported side by side for every arm, and **every
behavioural threshold in §2.5 is stated against the seed band, none against the within-run
interval.** A difference that clears the first but not the second is not a result.

### 2.4 Confounds and limitations

| # | Confound | Severity | Assessment / mitigation |
|---|---|---|---|
| **C1** | **Single seed per cell.** | **Critical** | Not mitigated — it is the design. §2.3 states the resolution bands; §4.4 names the confirmation runs. Every verdict in §7-§9 must carry the n = 1 qualifier. |
| **C2** | **ALL and X differ by 8 dimensions, not 2** (both interoceptive channels *plus* six proprioception channels), because proprioception is deliberately in neither restricted slice. | **Critical for the ALL-vs-X reading** | Not mitigable within this design; it follows from the user's recorded decision. **The pre-registered rule: no ALL-vs-X difference is ever attributed to interoception.** The interoceptive contrast is I vs X, and only that one. |
| **C3** | **Input width is not matched across the input factor** (27 / 2 / 19). Any I-vs-X difference confounds *which* information the modulator reads with *how much*, and with the size of its input layer. | **High for H5** | Not mitigable with a name-keyed sensor selector. Stated in every H5 verdict. A width-matched follow-up is available and named in §4.4: `Extero Nociception` alone (1 dim) against `Satiation` + `Interoceptive Nociception` (2 dims). |
| **C4** | **Pinning `rnn_mechanism: activation` breaks continuity with the historical neuromodulated record**, every run of which used the legacy gate-bias operator on the memory site. | Medium | Deliberate. Uniformity across sites is what makes Factor 2 a single factor; continuity with a record that confounds site with mechanism is worth less. The trade is recorded so a future reader does not read cell `t3rnn` as a replication of past memory-site runs. |
| **C5** | **Learning rates.** Today one optimiser trains the shared trunk, both heads and the modulator at `lr_actor`, and the advertised `lr_critic` is read by nobody. If [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]] lands first, `lr_critic` and a new `lr_modulator` become live. | **Critical if unhandled** | **Pre-launch requirement**: all 16 configs must set `lr_actor = lr_critic = lr_modulator = 0.0005`, **cell 1 included**. `lr_critic` currently has no consumer, so this changes nothing about how these runs train today; it is a guard against the fix landing later and a rerun from cell 1's saved config silently training the critic at one fifth of the rate the rest of the grid used. It is the **one deliberate deviation** from byte-identity with `recurrent_ppo_cmp_mc.yaml` (which carries `lr_critic: 0.0001`) and is recorded as such in §3.1 requirement 5. Verified by `env-config-reviewer` before launch. (`plan-reviewer` finding 4.) **Status 2026-09-07**: `lr_actor` and `lr_critic` are both `0.0005` in all 16 written configs. **`lr_modulator` is deliberately absent** — no such key exists anywhere in the code today, and writing a key no loader reads would be inventing schema. When the learning-rate fix does land and introduces it, these configs must be revisited before any rerun from them. |
| **C6** | **The meaning of `return_mode: MC` may change under the pending fix.** That fix's Part 1 proposes replacing the MC branch with "raw critic target, normalised advantage" — which is exactly the `MC_FIXED` mode that the later 25-run study measured as roughly an order of magnitude slower and 70 steps worse at matched experience. | ~~Critical, pre-launch blocker~~ → **RESOLVED 2026-09-07** | The two documents have now been reconciled, in this design's favour: the proposed change was implemented as the separate `MC_FIXED` mode, run at 5 seeds, and lost decisively, so [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]] now carries a do-not-apply banner on it. `return_mode: MC` keeps the semantics assumed throughout this design, the 10M-episode budget stands, and comparability with the existing five-seed `MC` record is preserved. Verified from the code as well as the docs: nothing under `src/` has changed since commit `e1aab726`. |
| **C7** | **Continuity of cell 1 with the five existing `MC` seeds** depends on no change to the training path between commit `788e5983` (those runs) and launch. | Medium | The site refactor promises the unmodulated path is bit-identical (its checks V1/C1). The pending fix does **not** — it changes loss values by design. **Pre-registered rule**: the five existing seeds are used for **dispersion** (how much seeds vary) and never for **level** (what the baseline is) unless the training path is confirmed unchanged. **Pre-registered sanity gate (the "C7 gate")**: cell 1 runs at seed 42 on the same environment, same agent config and same budget as the existing run `20260904-173804_rppo_cmp10m_mc_s42`, so it must **replicate** it — see §4.1 for the exact tolerance. If the unmodulated control does not reproduce the known result, the code path changed underneath the grid and **every** comparison in it is suspect, so the gate is checked first, before any hypothesis is scored. |
| **C8** | **Budget is in episodes, not environment steps, and better agents run longer episodes.** In the prior study total experience differed 3.3x across arms at a fixed episode budget. | **Critical** | Handled by design: the primary comparison is made at **matched environment steps** (the trainer logs `timesteps` directly), with the episode-axis number reported alongside. See §4.1. |
| **C9** | **Right-censoring.** Some arms may not have plateaued at 10M episodes. | High | Pre-registered exclusion rule in §5, item 4. |
| **C10** | **Checkpoints are not interchangeable across input arms.** Each slice changes the modulator GRU's input width, so the parameter trees differ. | Low — stated, not a defect | These are 16 fully independent trainings. No warm-starting, no shared initialisation beyond the common seed, no cross-arm checkpoint restore. Any attempt to restore across arms will fail loudly (the restore-completeness assertion), which is the desired behaviour. |
| **C11** | **The offline behavioural measures use a deterministic (argmax) policy** while the training-log survival series reflects the stochastic policy. They measure different objects. | Medium | Both reported side by side, never differenced. Same discipline as the prior study's C8. |
| **C13** | **The `Eat` action index cannot be verified from the trajectory store.** B0's denominator excludes `Rest` and `Eat`; only `Rest` has a witness column. `Eat` is hard-coded as 5 from the source, and `ate_food` is `False` when `Eat` is chosen on an empty cell, so it cannot confirm the index. | Low — stated, not mitigable from the store | The guard that *is* applied and *does* protect B0: **no row may carry a movement action and the `rested` column at the same time**, verified true on all five baseline runs and re-run on every grid store. If the action ordering ever changed, `Rest` would fail loudly and `Eat` silently. All 16 arms share one environment config with the baselines, which bounds the risk. See §4.2. |
| **C14** | **`env_fp` is not a same-world check for this study.** It hashes the whole resolved config including `seed`, `tag` and the wandb fields, so it differs across runs whose environment is identical — it already differs across all five baseline stores, which *are* the same world. | Medium — a **false-negative** hazard, not a false-positive one | Pairing is verified **empirically** instead: over 5,000 shared episode seeds the animal draws, observation draws, starting injury and spawn cell are bit-identical across all five baseline stores. Every grid arm will carry a distinct `env_fp` for the same reason, so a reader who treats it as a pairing check will wrongly conclude the comparison is invalid. Stated in §3 and in both collection specs. |
| **C12** | **`percept_add_bias_init` is read with a fallback default** (`recurrent_ppo_network.py:361` uses `.get(..., 0.0)`), contrary to the project's no-fallback rule. | Low | Pre-existing and out of scope for this design. Mitigated here by setting the key explicitly in all 15 modulated configs so no run depends on the fallback. Flagged to `bug-curator` in the handoff. |

### 2.5 Pre-registered predictions

Stated in advance, with the refutation criterion for each. Where the honest answer is "no
directional prediction", that is said rather than a direction being invented.

**A scoring rule that applies to H1, H2, H3 and H4 alike, fixed here in advance.** Every
target-versus-control comparison is scored **at the ALL input slice as the primary reading**, with
the best-performing input slice reported as a clearly-labelled secondary. Taking the best of three
input slices for a modulated target while the control has only one draw is a max-of-three selection
against an unselected comparator; in a fragile regime — where this design's own §2.3 lesson 2 puts
seed dispersion at 20-100 steps — that selection is worth tens of steps on its own and would
manufacture a win out of noise. ALL is also the status quo every historical neuromodulated run in
this project used, so it is the honest single slice to fix in advance. (`plan-reviewer` finding 5.)

**H1 — the write site matters at all.**
- *Confirmed if*: the range of survival across the six target conditions **at the ALL input slice**
  exceeds 30 steps at matched environment experience.
- *Refuted if*: all six target conditions fall within 15 steps of one another.
- *Prior*: weak. The project has no controlled site comparison; this is the first.

**H2 — the sensory front-end is the right place to act.**
- *Confirmed if*: **at the ALL slice**, `t2enc` beats `t1none` by more than 30 steps **and** is not
  itself beaten by `t3rnn`, `t4act` or `t5crt` by more than 30 steps.
- *Refuted if*: **at the ALL slice**, `t2enc` fails to beat `t1none` by 15 steps while some other
  single-site target does.
- *Secondary, labelled as such*: the same two comparisons taken at each target's best-performing
  input slice. A conclusion that holds only in the secondary reading is reported as "best-slice
  only" and is never a confirmation.
- *Prior*: this project's Paper 1 has already committed to the front-end injection site as its one
  shared modulator architecture. **That commitment has never been tested against the alternatives**,
  which is precisely why this cell matters. A refutation here is a finding about the paper's plan,
  not a bad result.

**H3 — re-tuning the value estimator helps. Directional prediction: positive.**
- *Confirmed if*: **at the ALL slice**, `t5crt` beats `t1none` by more than 30 steps.
- *Refuted if*: **at the ALL slice**, `t5crt` fails to beat `t1none` by 15 steps, or is worse.
- *Secondary, labelled as such*: the same comparison at `t5crt`'s best-performing input slice.
- *Prior, and why this is the one cell with an outside directional prediction*: the closest
  published precedent (a quadruped controller called PAPL) modulates every layer of both the action
  and value heads, and states the rule that the value head should be modulated **when the reward is
  itself conditioned on the modulating variable**. That precondition is met here and more strongly
  than a merely additive injury term would give: the reward is the step-to-step reduction in
  homeostatic drive, injury is one of the drive's two coordinates, and the marginal value of
  healing therefore rises with injury *and* depends on hunger. Separately, the only controlled
  actor-versus-critic comparison in this project's literature corpus (a fixed-wing aircraft study,
  Marquis & Farhood) found that FiLM-conditioning the value head **cut tracking error by roughly
  42-57% across four metrics**, while conditioning it with a different mechanism (LoRA) roughly
  doubled the error.
- *Two caveats carried forward with the prediction, both from the refactor plan's evidence review*:
  PAPL's conditioning variable is an **open-loop clock** — perfectly predictable — whereas ours is
  **contingent sensed injury** arriving through a 3-step smoothing kernel with no instantaneous
  leak, so our modulator cannot react on the step the damage lands; and the sign of value-head
  conditioning is **mechanism-dependent**, so the positive evidence transfers only because the
  mechanism here is FiLM.
- *A third caveat, added 2026-09-07 on `plan-reviewer` finding 9, and the most damaging of the
  three*: **in both cited papers the modulator reads information the main network never gets.**
  Marquis & Farhood hand the policy a 34-dimensional state and give the 6-dimensional fault vector
  **only** to the hypernetwork; PAPL's clock is likewise a side channel. In this grid the modulator
  reads a strict **subset** of what the main network already receives — under the ALL slice, exactly
  the same 27 numbers. Those results therefore support the proposition *"a critic conditioned on
  **extra** information helps"*, which is **not** the proposition this experiment tests. The
  directional prediction for H3 is kept, because the mechanism (FiLM) and the reward precondition
  (injury is an argument of the reward) both match, but it is now held **weakly**: a null at
  `t5crt` must not be written up as contradicting Marquis or PAPL, because it is consistent with
  both.
- *A free pre-check, requested before launch* (`plan-reviewer` finding 9, from the corpus's own
  recommendation): **plot the distribution of critic values conditioned on injury** on the five
  existing `MC` checkpoints. If the value function does not separate across injury levels, a
  critic-site modulator has nothing to condition on. This needs a small script — the critic's value
  output is computed inside `scripts/eval/eval_rollout.py` and discarded, and the trajectory store
  does not record it — so it is requested in §6.2 rather than assumed available.
  **Pre-registered ruling, either way**: the pre-check **does not veto** cell `t5crt` and does not
  move any H3 threshold. If values do not separate, an H3 null is reported as *expected* rather than
  as a surprise; if they do separate, an H3 null is the more informative result. If the pre-check
  cannot be run before launch, that fact is recorded and H3 is scored unchanged.

**H4 — more sites is better. No directional prediction.**
- Adding sites adds capacity, and each new site is initialised as a no-op (gain 1, offset 0), so at
  step 0 the all-four agent is exactly the single-site agent. That argument says adding sites cannot
  hurt *at initialisation*; it says nothing about the trained outcome, where four simultaneously
  adapting gain fields on a shared recurrent trunk could interact badly.
- *Recorded in advance*: `t16quad` landing below the best single-site target is **not** a surprise
  and is **not** evidence against modulation. Both directions are pre-accepted.

**H5 — the modulator must read the body. Two parts, scored in this order.** Directional prediction
for both. *Primary evidence is behavioural, not survival* (§4.2); the quantity compared is the
headline measure **B0**, the bush-entry rate broken down by starting injury.

**The reference every H5 number is scored against, fixed here: the seed band, never the within-run
interval.** Each arm's `Δ_B0` carries a very narrow within-run confidence interval — about
**±0.10 percentage points**, because 300,000 episodes characterise one agent precisely — and a
five-seed control band roughly **six times wider** (0.63 points in the primary window; measured
values in §4.2). **An arm reading `Δ_B0 = +0.3` points would clear its own error bar comfortably and
still sit inside the null band**, i.e. inside the range that five runs of the *identical unmodulated
configuration* already span for no reason but the random seed. Notation used below, per
(window x predator-condition) cell: **`T`** is the **top** of the five-seed control band for `Δ_B0`
and **`W`** is that band's **width** (max − min).

**Primary cell for every H5 ruling**: `Δ_B0` in the **first 25 steps** with **no predator within 2
cells**. Reasons: it is the window in which the randomised wound is still strongly felt; it has the
largest denominator of the two predator-conditioned readings; and it isolates state-dependence from
threat-response, which is B3's separate job. The whole-episode and predator-near readings are always
reported alongside; where they disagree with the primary, the disagreement is reported and the
primary carries the ruling.

**Per-arm ruling — applies to any single cell of this grid, under any behavioural hypothesis.**

| Where that arm's `Δ_B0` falls | Ruling |
|---|---|
| At or below `T` | **No evidence** of added state-dependence. |
| Above `T`, up to `T + W` | **Candidate, unresolved.** Not an effect and not reportable as one. Under the null, a sixth draw from the control's own seed distribution exceeds the maximum of five with probability about **1 in 6** — so one cell clearing the band is close to what the seed alone delivers. |
| Above `T + W` | **Candidate real effect.** Goes to a five-seed confirmation (§4.4). Still never a claim at n = 1. |

**No single cell of this grid can confirm a behavioural hypothesis. That is stated here, in the same
place as the thresholds, rather than discovered afterwards.** The only interpretable behavioural
readings this design produces are the **counts across the five write targets** defined below; a
per-cell number is a candidate, never a result.

**The smallest effect this grid can see**, stated so that a null is readable: one band width — about
**0.6 percentage points** of `Δ_B0` in the primary window, against a base rate near 7%, i.e. roughly
a **9% relative** change in how often a step out of the open lands in cover. (The whole-episode
pooled reading has a narrower band, 0.26 points, but a weaker treatment contrast.) **A refutation
below therefore means "no effect larger than about 0.6 points", not "no effect at all".**

**H5-add (scored first — this is the load-bearing one).** *Directional prediction: an
interoception-reading modulator produces more injury-dependence of cover-seeking than no modulator
at all.* Scored by **counting, across the five write targets at the I input slice, how many have
`Δ_B0 > T`** in the primary cell:

- *Confirmed if*: **3 or more of the 5.** Under the null — every I arm behaving as a fresh draw from
  the control's own seed distribution — that count arises with probability about **3.5%**. Two
  sentences travel with that figure wherever it is reported: it assumes the five targets are
  independent draws, and they are not quite — they share seed 42 and the identical paired episode
  population, which correlates them positively and therefore **inflates** the true false-positive
  rate above 3.5%. It is a screening-level signal, not a p-value.
- *Unresolved if*: **exactly 2 of the 5** (null probability about 16%).
- *Refuted if*: **1 or 0 of the 5.** This is the *expected* outcome if the modulator adds nothing
  (null probability about 80%), which is precisely what makes it a usable refutation criterion. It
  says a body-reading modulator adds no cover-seeking state-dependence beyond what the main network
  — which receives both body channels in every arm, the control included — already produces on its
  own. Given the pre-launch finding that the control produces **none, and if anything slightly
  negative** (§4.2), such a refutation says the modulator fails to **create** the behaviour, not
  that it fails to amplify it. It is the single most consequential negative this grid can return and
  must be reported as such rather than folded into an "inconclusive" summary.
- *Untested assumption carried with all three rulings, recorded as **A3-B***: that a **modulated**
  arm's seed dispersion is no wider than the unmodulated control's. Nothing in this project has
  measured that. If modulated arms are behaviourally noisier, every probability above is optimistic.
- *Why this is the load-bearing test* (`plan-reviewer` finding 7): H5-spec below can be satisfied
  by a modulator that changes behaviour at all, because X's modulator cannot read satiation and
  reaches injury only indirectly. Only the comparison against the **unmodulated** control asks
  whether the modulator is contributing the state-dependence rather than merely correlating with it.

**H5-spec (the specificity check, scored second).** *Directional prediction: I exceeds X.* The two
arms compared are each a single run, so neither carries a band of its own; the dispersion of a
*difference* between two independent single draws is wider than the dispersion of one draw by
roughly a factor of √2. The gate is therefore set at one full band width — **`I − X > W`** — which,
under a normal approximation to the five-seed band, is about **1.65 standard deviations** of that
difference, i.e. a per-target false-positive rate near **5%**.

- *Confirmed if*: **3 or more of the 5** write targets satisfy `I − X > W` (null probability about
  0.1%).
- *Flagged, unresolved if*: **exactly 2 of the 5** (null probability about 2%).
- *Refuted if*: **1 or 0 of the 5** — and this ruling is registered together with the reason it is
  weak: the null delivers that outcome about **98%** of the time, so **an H5-spec refutation is
  nearly uninformative** and may **not** be written up as evidence that the modulator ignores the
  body. Only an H5-spec confirmation carries information, and even then only as a screen.
- *Reported with its own caveat, always*: this contrast is close to true by construction, so a
  confirmation here **without** a matching H5-add confirmation is written up as "consistent with,
  but not evidence for, the mechanism".

**Direction counts; magnitude alone never does.** Every threshold above is one-sided in the
**positive** direction, because the project's target behaviour is *more* cover-seeking when wounded.
An arm whose `|Δ_B0|` is large but **negative** is reported as "state dependence in the direction
opposite to the project's target behaviour" — a finding in its own right, the outcome the
freeze-to-heal record predicts, and what the five unmodulated controls actually do (§4.2).

*Not evidence for either part*: an ALL-vs-X difference (confound C2); or an I-vs-X difference in
**survival** alone with no matching behavioural difference (a modulator can raise survival by
statically route-tuning the encoder while producing no context-dependent policy at all).

**H6 — restricting the input costs survival. No directional prediction; registered as a rival
reading.**
- *The check*: if survival falls monotonically with input width (ALL > X > I) at **four or more** of
  the five write targets, the input factor is behaving like a capacity/information effect, and any
  I-vs-X survival difference must be read that way rather than as an interoception effect.
- This check runs **before** H5 is scored, and its outcome is reported whether or not H5 is
  supported.

**Cells with no prediction at all**: `t3rnn` (memory) and `t4act` (action head) against the control.
Neither has a precedent in the project's corpus in this form, and inventing a direction for them
would be a fishing expedition. They are here to be measured.

### 2.6 Compute estimate

The unmodulated `MC` arm of the return-mode study ran 10M episodes (about 1,516 M environment
steps) in roughly **10.8 hours** of wall clock (run directory `20260904-173804_rppo_cmp10m_mc_s42`,
first write 2026-09-04T17:38, last write 2026-09-05T04:28), on a lab GPU shared with other runs of
the same batch. The modulated arms add a 16-unit GRU and a handful of FiLM heads; the refactor's own
speed check expects the cost to be small but has not yet reported a number.

**Estimate: 11-15 hours per run.** Nodes 106-114 offer 20 GPU slots (106-112: two RTX 3090 each;
113: two RTX 4090; 114: four RTX 6000 Ada), so all 16 runs fit in a single wave at one run per GPU,
giving roughly **12-16 hours wall clock for the whole grid**. The prior batch packed several runs
per node, so tighter packing is feasible; placement is `training-runner`'s call against the live
GPU state, following the pack-node-first policy.

### 2.7 Dependencies — all three now cleared

The design was written before the code it needs existed, and deliberately stopped short of
producing YAML until three things landed. **All three have.** The table is kept rather than
deleted, because which commit each arm depends on is part of the record.

| # | Dependency | State at time of writing | Blocks |
|---|---|---|---|
| **D-A** | **Part A of the site refactor** — the `agent.modulation.sites.{encoder,rnn,actor,critic}`, `agent.modulation.rnn_mechanism` and `agent.modulation.temperature.{enabled,clip}` keys, plus FiLM at the action and value heads. | **CLEARED 2026-09-07** — landed in commit `83b8140b`. All four site keys are individually mandatory; a config with every site off *and* temperature off is refused outright. | Factor 2 entirely. |
| **D-B** | **Part B of the site refactor** — the `agent.modulation.input_sensors` key (`"all"` or an explicit list of sensor names, resolved to indices against the run's own observation breakdown). | **CLEARED 2026-09-07** — landed in commit `e1aab726`. Mandatory whenever `modulation.type` is non-null; an unknown sensor name is a hard error rather than a silent re-indexing. | Factor 1 entirely. |
| **D-C** | **A ruling on `return_mode: MC` semantics** — see confound C6. | **CLEARED 2026-09-07.** The proposed change to the MC branch was built as the separate `MC_FIXED` mode, run at scale, and lost decisively (25 runs; raw-target arms reached roughly 67–98 survival steps against 156–160 for the z-scored-target arms at matched experience). [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]] now carries a do-not-apply banner for it, so `return_mode: MC` keeps the semantics this design assumed and the 10M-episode budget stands. | The whole grid's budget and its comparability with the existing 5-seed baseline. |

All three are cleared, and the configs were produced in the second pass on 2026-09-07. §3.1 below
records what was written and what was checked.

---

## 3. Launch Manifest

System-of-record for every run. `experiment-designer` owns the planned columns below;
`training-runner` fills Node / GPU / Launched at / WandB run ID / Log path in place at launch;
`experiment-analyzer` reads the table to find the runs. **The runner does not invent tags for runs
in this manifest** — the values below are authoritative.

Tag scheme: `rppo_nmnsite_<target-slug>_<input-code>_s<seed>`. The stem `rppo_nmnsite_` was checked
against all 387 existing run directories under `results/JAX_RecurrentPPO/` and collides with none.
(The near-miss `rppo_nmn_t*` was rejected because it is also a prefix of the existing
`rppo_nmn_tempceil*` family, which would break grep-based grouping.) Tag and wandb-name are
identical on every row.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Code SHA | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|----------|------|-----|-------------|--------------|----------|
| 1 | planned | `T1_none` | `rppo_nmnsite_t1none_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 2 | planned | `T2_enc_ALL` | `rppo_nmnsite_t2enc_ALL_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 3 | planned | `T2_enc_I` | `rppo_nmnsite_t2enc_I_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 4 | planned | `T2_enc_X` | `rppo_nmnsite_t2enc_X_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 5 | planned | `T3_rnn_ALL` | `rppo_nmnsite_t3rnn_ALL_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 6 | planned | `T3_rnn_I` | `rppo_nmnsite_t3rnn_I_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 7 | planned | `T3_rnn_X` | `rppo_nmnsite_t3rnn_X_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 8 | planned | `T4_act_ALL` | `rppo_nmnsite_t4act_ALL_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 9 | planned | `T4_act_I` | `rppo_nmnsite_t4act_I_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 10 | planned | `T4_act_X` | `rppo_nmnsite_t4act_X_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 11 | planned | `T5_crt_ALL` | `rppo_nmnsite_t5crt_ALL_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 12 | planned | `T5_crt_I` | `rppo_nmnsite_t5crt_I_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 13 | planned | `T5_crt_X` | `rppo_nmnsite_t5crt_X_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 14 | planned | `T16_quad_ALL` | `rppo_nmnsite_t16quad_ALL_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 15 | planned | `T16_quad_I` | `rppo_nmnsite_t16quad_I_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |
| 16 | planned | `T16_quad_X` | `rppo_nmnsite_t16quad_X_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — | — |

**`Code SHA`** is filled by `training-runner` from each run's own `provenance.json`, not from a
`git rev-parse` at launch time. All 16 rows must show the **same** SHA with `git_dirty: false`; a
row that does not is a run that cannot be reconstructed and is excluded from every comparison. Any
relaunch under §5 item 1 checks out that same SHA in a worktree. (`plan-reviewer` finding 2, which
is the launching agent's to clear — the column is here so there is somewhere to record the answer.)

**What that SHA has to contain, fixed here so the runner can check it in one command.** The
training code this grid depends on is the modulation-site refactor: its Part A (selectable FiLM
sites) is commit `83b8140b` and its Part B (the modulator input slice) is commit `e1aab726`, both
of 2026-09-07. **Nothing under `src/` has changed since `e1aab726`**, and the sixteen agent
configuration files were committed on top of it as `03e590a4`. So whatever HEAD is at launch, it
must (a) have `e1aab726` as an ancestor, (b) show no difference under `src/` against it, and (c) be
the same on all 16 rows with a clean working tree. A grid launched from two different SHAs is not a
controlled comparison, whatever the diff between them turns out to be.

`wandb-job-type` is **`pilot`**, not `prod`, on every row. That is the honest label for a
single-seed screen and it keeps these runs from being pooled with production multi-seed studies in
any downstream query.

**How these runs are paired for the behavioural comparison, and the check that does *not* work.**
Every run in this manifest is evaluated afterwards on the same 300,000 episode seeds
(`seed_base` 1,000,000 — the trajectory-collection spec named in §3.3), which is what makes every
arm-to-arm behavioural comparison **paired**: episode *i* presents the same world to every agent, so
a difference between two arms cannot be a difference in the worlds they were shown. The obvious way
to *verify* that would be the `env_fp` fingerprint the trajectory collector stamps into each store's
manifest — and **it does not work here.** `env_fp` hashes the whole resolved configuration,
including `seed`, `tag` and the WandB fields, so it differs between two runs whose environment is
identical. It already differs across all five unmodulated baseline stores, which *are* the same
world, and it will differ across all 16 grid arms for exactly the same reason. **Anyone using
`env_fp` as a same-world check on this experiment gets a false negative.** Pairing on the five
baselines was therefore verified empirically instead: over 5,000 shared episode seeds the animal
draws, the observation draws, the starting injury and the spawn cell are **bit-identical** across
all five stores. That direct check — not `env_fp` — is the one to repeat on the grid arms.

### 3.1 Configs — WRITTEN AND VERIFIED (2026-09-07, commit `03e590a4`)

All 16 runs share one environment config, unmodified. Each run gets its own agent config, because
agent configs in this repo do not support `extends:` and are self-contained.

**In plain language, what exists now.** All sixteen agent configuration files have been written.
They were not typed out sixteen times: a small program in the same folder produces them, and it
builds each one by reading the shared settings straight out of the already-established unmodulated
agent file and then changing only the two things this study varies. Every one of the sixteen was
then loaded the way the trainer loads it and used to build a real network, which is how the claims
below about "the modulator reads 2 numbers here and 19 there" were established — they were read off
the constructed network, not worked out on paper.

| Run | Config (env) | Config (agent) — written, verified |
|-----|--------------|--------------------------------|
| 1 | `configs/environment/experiment/basic/04-jump_attack_10x10.yaml` | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t1none.yaml` |
| 2 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t2enc_ALL.yaml` |
| 3 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t2enc_I.yaml` |
| 4 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t2enc_X.yaml` |
| 5 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t3rnn_ALL.yaml` |
| 6 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t3rnn_I.yaml` |
| 7 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t3rnn_X.yaml` |
| 8 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t4act_ALL.yaml` |
| 9 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t4act_I.yaml` |
| 10 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t4act_X.yaml` |
| 11 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t5crt_ALL.yaml` |
| 12 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t5crt_I.yaml` |
| 13 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t5crt_X.yaml` |
| 14 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t16quad_ALL.yaml` |
| 15 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t16quad_I.yaml` |
| 16 | same | `configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t16quad_X.yaml` |

**Requirements the second pass must satisfy**, recorded now so they cannot be forgotten:

1. **Generated, not hand-copied.** Sixteen near-identical 60-line files differing in two blocks is
   a drift hazard. The pass writes a small generator alongside them, following the precedent of
   `configs/environment/experiment/sensory_directional/generate_weakened_vision_arms.py`. The
   generator lives under `configs/`, not `scripts/`, so it does not trigger the scripts
   dependency-map maintenance contract.
2. **Explicit learning rates in every file** — `lr_actor`, `lr_critic` and (if the key exists by
   then) `lr_modulator`, all at `0.0005`, in **all 16** files including cell 1. See C5. `lr_critic`
   has no consumer today, so this is inert for these runs; it exists so that a later rerun from any
   of these saved configs, after the learning-rate fix lands, trains every arm's critic at the same
   rate. (`plan-reviewer` finding 4.)
3. **`percept_add_bias_init: 0.0` stated explicitly** in all 15 modulated files, so no run depends
   on the fallback default noted in C12.
4. **`memory_bias_init` and `memory_clip` present** in all 15 modulated files even though they are
   unused under `rnn_mechanism: activation` — the refactor keeps them mandatory.
5. **Run 1 carries `modulation: {type: null}`** and no other modulation keys. Under
   `train.py:1138-1140` a null type collapses to no modulator at all, so none of the new mandatory
   keys is read. Its agent keys must be identical to
   `configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml` **except for `lr_critic`**, which that
   file sets to `0.0001` and which this grid sets to `0.0005` everywhere per requirement 2. That is
   the **only** permitted difference, it changes nothing about how Run 1 trains today (`lr_critic`
   is read by no code path), and it is recorded here because requirements 2 and 5 as originally
   written could not both be satisfied. The lineage link to the existing 5-seed record is otherwise
   exact (subject to C7 and its gate in §4.1). (`plan-reviewer` finding 4.)
6. **`env-config-reviewer` runs on all 16 files before launch**, with explicit attention to C5
   (learning rates), the input-sensor name spellings against
   `get_observation_breakdown`, and the sites/mechanism/temperature block.

#### How each requirement was met (recorded 2026-09-07)

| # | Requirement | How it was satisfied |
|---|---|---|
| 1 | Generated, not hand-copied | `configs/models/recurrent_ppo/nmn_input_site_grid/generate_site_grid_arms.py`, committed alongside the sixteen files. It lives under `configs/` rather than `scripts/` so it does not trigger the scripts dependency-map contract, following the precedent generator in the sensory-directional sweep. `--check` re-derives every file and fails on any byte of drift; `--verify` adds the model-construction pass below. |
| 2 | Explicit learning rates in all 16 | `lr_actor: 0.0005` and `lr_critic: 0.0005` are present in every file, control included. **`lr_modulator` was NOT added**: no such key exists anywhere in the code today (`grep` over `src/`, `train.py` and `configs/` returns nothing), and writing a key no loader reads would be inventing schema, which this design forbids. The requirement's own wording made it conditional on the key existing. |
| 3 | `percept_add_bias_init: 0.0` stated explicitly | Present in all 15 modulated files, so no run depends on the fallback default flagged as confound C12. |
| 4 | `memory_bias_init` and `memory_clip` present | Present in all 15 modulated files, though inert under `rnn_mechanism: activation`. |
| 5 | Control identical to its ancestor except `lr_critic` | True **by construction**, not by inspection: the generator loads `configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml` and mutates exactly that one key. Independently re-checked as a flattened key-by-key comparison — the only difference found is `agent.lr_critic`. |
| 6 | `env-config-reviewer` before launch | **Outstanding.** The one requirement still open; see the handoff note at the end of this section. |

#### What was verified, and how

Everything below was produced by `generate_site_grid_arms.py --verify`, which loads each config
through the same merge order the trainer uses and then **constructs the model**. A file that merely
parses as YAML proves nothing here: the point of the refactor's mandatory keys is that a wrong
config is refused loudly, so the refusal is collected now rather than at launch.

- **All sixteen load, construct and take a step.** Every arm builds a model and returns finite
  action scores and a finite value estimate on a forward pass.
- **The modulator's input width was read off the built network**, not computed by hand: 27 numbers
  under ALL, 2 under I, 19 under X — confirmed three ways per arm (the resolved index tuple, the
  modulator GRU's declared input width, and the actual row count of its input weight matrix).
- **The four site switches were read off both the network and the modulator inside it**, so a
  copy-paste that left two sites on would have been caught. Every arm resolves to exactly the
  combination §2.1 specifies.
- **The environment is the same for all sixteen.** Each arm's observation layout was recomputed and
  compared: `Satiation 1, Interoceptive Nociception 1, Extero Nociception 1, Olfaction 5,
  Collision 5, Proprioception 6, Visual 8 = 27` in every case. The sensor-name spellings in the
  restricted slices are exactly these keys.
- **Config-to-config differences are only the intended ones.** Flattened key-by-key: the control
  differs from its ancestor `recurrent_ppo_cmp_mc.yaml` in `agent.lr_critic` and nothing else; each
  modulated arm differs from the reference arm (`nmnsite_t2enc_ALL.yaml`) only in the four
  `sites.*` switches and `input_sensors`.
- **The mechanism controls hold.** Every modulated arm resolves to the uniform FiLM operator, the
  `activation` RNN mechanism, temperature **off**, `grouping_size: 1`, `mod_hidden_size: 16`, and —
  importantly for confound C4's flip side — the **same plain GRU cell the unmodulated control uses**,
  not the legacy modulated cell.
- **Negative controls: the loud failures really are loud.** A deliberately misspelled sensor name, a
  missing site key, and all-sites-off-with-temperature-off each raise an error at construction
  rather than building a plausible-looking wrong model.

Re-run at any time (CPU only; it never competes with training for a GPU):

```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
  configs/models/recurrent_ppo/nmn_input_site_grid/generate_site_grid_arms.py --verify
```

#### Still outstanding before launch

- **`env-config-reviewer` pre-flight** on all sixteen files (requirement 6). Points to put in front
  of it: the learning-rate deviation of C5; the sensor-name spellings against
  `get_observation_breakdown`; the sites / `rnn_mechanism` / temperature block; the deliberate
  absence of `temperature.clip` when temperature is disabled; and the deliberate absence of
  `lr_modulator`.
- **The trajectory-collection spec for the arms cannot be completed yet.**
  `configs/trajectory_collection/nmn_site_grid_arms.yaml` still carries its `<RUNDIR_*>`
  placeholders, and it has to: each entry needs a run **directory** name, and those are stamped with
  the launch timestamp, so they do not exist until the runs do. The labels are already final and
  match the Cell column of the manifest; only the timestamps are missing. Fill them after launch,
  from the manifest's Log path column.

### 3.2 Launch command

One invocation per run, through `run_command.py` onto the assigned node. Shown for Run 11
(`T5_crt_ALL`); the other 15 differ only in `--agent_config`, `--tag`, `--wandb-name` and
`--device`.

```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/nmn_input_site_grid/nmnsite_t5crt_ALL.yaml \
  --episodes 10000000 \
  --device cuda:0 \
  --log-interval 10 \
  --tag "rppo_nmnsite_t5crt_ALL_s42" \
  --wandb-name "rppo_nmnsite_t5crt_ALL_s42" \
  --wandb-group "nmn_input_site_grid" \
  --wandb-job-type "pilot"
```

`--seed`, `--num-envs` and `--checkpoint-frequency` are **not** passed: they are config-owned
(`configs/train/default.yaml`, overridden by `configs/train/recurrent_ppo.yaml`) per the project's
config-owns-values convention.

### 3.3 Trajectory-collection specs (post-training evaluation population)

Training produces the runs; the **behavioural** measures of §4.2 are computed afterwards, by rolling
each finished agent out over a large fixed population of episodes and storing every step. That
collection is a scientific act with its own parameters, so it gets its own committed spec files
rather than a shell script — this study's baseline collection was driven ad hoc and its commands
survived only in a scratch file, which is the same gap the sensor-ladder study's own spec header
complains about.

| Spec | Covers | State |
|---|---|---|
| `configs/trajectory_collection/nmn_site_grid_baselines.yaml` | The five unmodulated `MC` runs that supply the control band (§4.2). | **Regenerated from the five stores already on disk.** Re-running it reproduces the population, not the original invocation. |
| `configs/trajectory_collection/nmn_site_grid_arms.yaml` | All 16 grid arms. | **Written; not runnable yet.** Every run path is a placeholder, because the run directory names do not exist until launch. The labels are final and match the Cell column of the manifest above. |

Both specs pin the same population parameters: **300,000 episodes**, **`seed_base` 1,000,000**,
`obs_precision: float32`, final checkpoint, deterministic (argmax) policy, one collector process per
node.

**The one value that must not drift is `seed_base: 1000000`.** It is what makes every arm-to-arm and
arm-to-baseline behavioural comparison **paired** — episode *i* presents the same world to every
agent. Collect an arm under any other base and the comparison silently degrades from paired to
unpaired: both collections succeed, no error is raised, no warning is printed, and the resulting
numbers look exactly as legitimate as paired ones. Nothing downstream can detect it after the fact
except by reading the two manifests. Preventing that specific silent failure is the reason these two
files exist.

**Three things about the spec schema, recorded so nobody improvises around them:**

- **`restore_check: strict` is not a spec key** — adding it is a hard error, because
  `run_collection.py` rejects unknown top-level keys by design. Strict restore checking is what the
  driver gives you *structurally*: the only way to weaken it is a flag on the collector that the
  driver has no path to pass. Every store records `restore_check` in its own manifest; check it
  there. This matters more than usual here, because ten of the sixteen arms carry a modulator whose
  input width differs from the others, so their parameter trees differ — the strict check is what
  turns a mismatched restore into a loud failure rather than a silently-unmodulated agent producing
  plausible-looking behaviour.
- **`device: gpu` means one process per node**, and `npar: 1` is stated explicitly in both specs as
  a correctness choice rather than a throughput one. The ad hoc baseline collection put three
  processes on two GPUs at batch size 5,000 and the third died out of memory. To go faster, add
  nodes, not processes per node.
- **A known gap in the schema, flagged rather than papered over.** §4.2 asks for two readings per
  run — final checkpoint and nearest-to-matched-budget. The second is **not expressible** in the
  current spec: `checkpoints` is one list applied to every run, but each arm's matched-budget
  checkpoint falls at a different episode-indexed step, because a better agent runs longer episodes
  and equal environment steps therefore means unequal episodes. That pass will need either a per-run
  checkpoint capability or one spec per step group. It is not improvisable with a per-run key —
  unknown per-run keys are a hard error, by design.

---

## 4. Analysis plan (pre-specified)

### 4.1 Primary outcome — survival steps

**Survival steps, never cumulative reward.** Two readings, both reported, never pooled:

1. **At matched environment experience** — the primary. Survival averaged over a trailing 2% window
   in *step space*, read at the largest environment-step budget reached by all 16 runs. The trainer
   logs `timesteps` directly, so no inversion of the episode curve is needed. This is the reading
   that carries every verdict, because at a fixed episode budget a better agent consumes more
   experience (confound C8) and the prior study showed that confound to be worth a factor of 3.3.
2. **At the end of the episode budget** — reported alongside for continuity with the existing
   record, and explicitly labelled as confounded.

**Statistic**: a single number per cell, with **no confidence interval across seeds, because there
is one seed.** Reporting a mean-plus-interval over one run would be a false precision and is
forbidden by this pre-registration. Where a within-run interval is meaningful (the behavioural
measures) it is reported and labelled as within-run.

**Effect-size rulings**: exactly the bands in §2.3. They are fixed and may not be moved after the
data is seen.

**Temporal evolution — mandatory, not optional.** For every run, survival against environment steps
read at 50 M, 100 M, 200 M, 400 M, 800 M and the common maximum, so that an arm that wins early and
plateaus is distinguishable from one that is still climbing. Two derived quantities per run:

- **Time to reach 100 survival steps** (environment steps consumed), a second and partly
  independent read of the same run, analogous to the prior study's escape-time measure.
- **Late-training slope**: change in survival over the final 20% of the episode budget. Feeds the
  right-censoring rule in §5.

**The C7 gate — scored before anything else, and it can void the grid.** Cell 1 is the same
environment, the same agent settings, the same budget and the same seed (42) as the completed run
`20260904-173804_rppo_cmp10m_mc_s42`, so it must **reproduce** it. Pre-registered check, in order:

1. **Trajectory check** — cell 1's survival-versus-environment-steps curve is overlaid on that run's.
2. **Level check** — cell 1's end-of-budget survival must fall inside the five existing `MC` seeds'
   range (162.4 to 166.9 steps). Not "close to the mean": inside the observed range.
3. **Ruling if it fails.** A cell-1 result outside that range means the training path changed
   between commit `788e5983` and launch, in a way that affects the *unmodulated* path. Every
   comparison in this grid is then suspect, because the control is the reference for all of them.
   The grid is reported as **void pending diagnosis**, not as a set of results with a caveat, and
   the discrepancy is routed to `senior-developer`. Nothing in §7-§9 is written until it is
   explained.
4. This gate is checked at the earliest checkpoint at which both curves exist, not only at the end,
   so a divergence is caught in the first hours rather than after 15 hours x 16 runs.

(`plan-reviewer` assumption A1; the review asked for exactly this to be in the analysis plan.)

### 4.2 Behavioural outcome — does the policy actually become context-dependent?

Survival alone cannot show context-dependent policy. A modulator can raise survival by statically
re-weighting sensory channels without ever conditioning behaviour on internal state. These measures
are what test the actual claim — and **this section was rewritten on 2026-09-07**, before any run
existed, because `plan-reviewer` (finding 3) showed that its first version had walked back into the
trap it was written to avoid.

**The trap, in plain language.** An earlier analysis ([[INJURY_HIDING_SIGN_RECONCILIATION]]) found
that a wounded agent's extra time in bushes is mostly **freezing to heal**, not travelling to cover.
The wounded agent stops moving; whether that registers as "more hiding" or "less hiding" depends on
whether it happened to be standing on a bush when it froze. Across fourteen agents, "hides more when
injured" was positive in 13 of 14 over the first 25 steps, in 2 of 14 over the first 100, and in
1 of 14 over the whole episode. Any measure built on **where the agent is** inherits that artefact.

**Why the first version of this section did not escape it.** It proposed, as the headline, a
bush-occupancy fraction computed only over steps where the agent was *not* resting, and scored it
against a reference band of −1.8 to −0.8 percentage points taken from the fourteen-agent record.
Three things are wrong with that, and all three were caught before launch:

1. **The band measures a different quantity.** It is the "in a bush *while acting*" column of a
   decomposition of total bush dwell over **all** steps (rest-in-bush plus act-in-bush equals dwell).
   It falls with injury partly **mechanically**, because acting itself falls by 18-32 percentage
   points. The measure proposed here was a fraction **of acting steps** — a different denominator,
   with an unknown baseline sign. The "flip toward zero or positive" threshold was therefore being
   scored against a null that does not belong to it.
2. **Conditioning on "not resting" does not remove the artefact.** Both the injury and the agent's
   position plausibly cause whether it rests, so a negative value can appear with no decision to
   hide less, and an agent that genuinely travels to cover more can still read negative.
3. **The same objection hits the threat-response range.** A frozen agent cannot answer a predator by
   moving, so the size of the "hides more when a predator is near" effect shrinks in the
   high-injury bins whether or not any context-dependent decision is being made.

**The escape is to measure a movement decision, not a location.** The headline measure below is
scored only on steps where the agent **chose to move**, and asks where that movement went. An agent
that freezes contributes nothing to it, in either direction — freezing removes the step from the
denominator instead of loading it into the numerator.

#### B0 — bush-entry rate (the headline behavioural measure)

> **In one sentence**: of the times the agent was standing in the open and decided to take a step,
> how often did that step land it in a concealing bush — and does that fraction rise when it starts
> the episode more badly wounded?

**Definition.** `B0 = P(in a bush at t+1 | not in a bush at t, and a movement action was chosen at
t)`, expressed in percentage points.

**Exact computation against the trajectory store**, written out because the store's row convention
is a known footgun (row `t` holds the environment state **at** `t` together with the action that
**arrived at** `t`, per [[TRAJECTORY_STORE_SCHEMA]] §3.1):

- *Denominator*: rows `t` with `agent_in_bush[t] == False` **and** `action[t+1] ∈ {0,1,2,3}` (the
  four moves). Both `Rest` and `Eat` are excluded — the denominator is "the agent decided to move",
  not "the agent was not resting".

  **A verification limit, recorded because it cannot be fixed from the store.** The `Rest` index
  *is* independently checkable per run: the store carries a `rested` column, and the guard actually
  applied to all five baseline runs is that **no row may carry a movement action and `rested` at the
  same time** — which held on every one of them. The `Eat` index has **no such witness**. It is
  hard-coded as **5** from `src/environment/core.py`, and no stored column identifies it: `ate_food`
  is `False` whenever `Eat` is chosen on an empty cell, so it cannot confirm the index. If the
  action ordering ever changed, the `Rest` exclusion would fail **loudly** and the `Eat` exclusion
  would fail **silently**. The mitigation available today is that movement-vs-`rested` guard,
  re-run on every store and its result reported, plus the fact that all 16 grid arms share one
  environment config with the five baselines. Stated so no future reader assumes the `Eat`
  exclusion was verified.
- *Numerator*: those same rows with `agent_in_bush[t+1] == True`.
- *Injury binning* — **RE-REGISTERED 2026-09-07**, before any grid run existed, because the
  previously registered edges described a contrast that **cannot occur**. Bins are formed on the
  **randomised starting injury** the environment drew before the agent acted — the only causally
  identified internal-state variable available.

  **What was wrong.** The first version of this section adopted the edges already implemented in
  `scripts/analysis/context_dependence.py` (`INJ_EDGES = [1e-9, 25, 50]`), giving bins *exactly 0* /
  *0–25* / *25–50* / *50+*, and defined `Δ_B0` as top-populated minus bottom-populated bin. But the
  environment draws starting injury from a **continuous uniform distribution over 0 to 100**
  (`src/environment/core.py:1103-1107`, bounds `start_injury_low: 0` and `start_injury_high: 100`
  set in `basic/03-random_init_10x10.yaml` and inherited by `basic/04`). A continuous draw is never
  *exactly* zero: across 15,000 sampled episodes the smallest starting injury observed was
  **0.0058**, with **zero** exact zeros. The "exactly 0" bin is therefore **structurally empty**;
  the minimum-count substitution rule fired on all five baseline runs; and every `Δ_B0` actually
  computed was *0–25* versus *≥50* — a narrower injury contrast than the document implied, taken
  against a bottom bin whose label reads "uninjured" and which is not.

  **The re-registered edges: `[25, 50, 75]` — four bins, 0–25 / 25–50 / 50–75 / 75–100.** `Δ_B0` is
  the **75–100 bin minus the 0–25 bin**. Three reasons, in order of weight:

  1. **No empty bin, and no bin that misdescribes itself.** Every bin is populated by construction
     and every label is honest about the injuries it contains.
  2. **They are fixed edges *and* the population quartiles at the same time.** Because the draw is
     uniform on 0–100, the quartile boundaries are the known constants 25 / 50 / 75, so these edges
     produce four equal-probability bins (about 75,000 of 300,000 episodes each) **without** being
     defined by the sample. That resolves the tension recorded in §10.1(b): `plan-reviewer` asked
     for quartiles; this design refused *sample-defined* quartiles because they would sit at
     different injury values in each of the sixteen arms and destroy comparability. Under a known
     uniform draw the two requirements coincide, so both are met.
  3. **A stronger treatment contrast, at a cost in precision that does not bind.** The top bin
     becomes *75–100* rather than *50–100*, roughly doubling the injury separation between the two
     compared bins. It also halves that bin's sample, widening the within-run binomial interval by
     about √2 — from roughly ±0.10 to roughly ±0.13 percentage points. That is irrelevant here,
     because §2.5 scores everything against the five-seed band (0.63 points), which is about five
     times wider than either figure. Buying effect size with precision is the right trade when
     precision is not the binding constraint.

  **The cost, stated rather than buried.** These are **not** the edges the fourteen-agent reference
  record ([[INJURY_HIDING_SIGN_RECONCILIATION]]) used. That record's numbers are therefore cited in
  this document for **sign and order of magnitude only** — never as a numeric null for anything
  scored here. Every number this grid is scored against comes from the five unmodulated baselines
  recomputed on the edges above (see "The control band as measured", below). All four measures
  B0–B3 use **one** set of edges, so no reader has to track two binnings.

  **Minimum bin size and substitution**, unchanged in rule and tightened in reporting: a bin needs
  at least 2,000 qualifying denominator steps; otherwise the next bin inward is taken. With about
  75,000 episodes per bin the rule is not expected to fire at all, and if it fires on any arm that
  is itself reportable — it would mean that arm almost never steps out of the open. Realised bin
  counts are reported for every arm.
- *Predator condition*: at row `t`, `near` = any **active predator** within Chebyshev distance 2 of
  the agent, matching `NEAR_D = 2` in the same script. Reported separately for `near` and `far`.
- *Windows, both always reported*: the **first 25 steps** (`t <= 25`, the window in which the
  randomised wound is still strongly felt, and the window the fourteen-agent record uses) and the
  **whole episode**. Both bin by the randomised starting draw, so both are causally identified; the
  early window has the stronger treatment and the whole-episode window the larger sample.
- *Robustness variant, reported alongside*: the same measure with the denominator conditioned on a
  **realised displacement** (`(agent_row, agent_col)` differs between `t` and `t+1`) instead of on
  the chosen action. This drops moves that a rock blocked. The two variants differ only through
  collisions; a disagreement between them is itself reportable.
- *Secondary variant, only if it can be derived cleanly*: the same rate with the denominator further
  restricted to steps where **cover was reachable in one move** (at least one of the four adjacent
  cells is an active concealing obstacle). The per-slot concealment flag is **not** a stored column;
  it must be derived from the run's own saved environment config (`hides_agent` per expanded
  obstacle entry, `config_loader.py:1249`) joined with the store's `obs_active` and `obs_row` /
  `obs_col`. If that derivation cannot be validated, this variant is **dropped rather than
  approximated** — the two primary variants above stand on their own.

**The statistic that carries H5.** For each arm, window and predator condition:

`Δ_B0 = B0(75–100 injury bin) − B0(0–25 injury bin)`, in percentage points, under the re-registered
edges above.

The two bins are **named explicitly** rather than as "top populated" and "bottom populated". That
wording is exactly what let an empty bin change the contrast silently in the first version of this
design, and it is not used again. If the 2,000-step minimum-count rule does fire, the substitution
is reported **in the same table cell as the number**, never in a footnote.

**Direction is pre-registered, and magnitude alone does not count.** The project's target behaviour
is *more* cover-seeking when wounded, so the predicted sign of `Δ_B0` is **positive**. An arm whose
`Δ_B0` is large but **negative** is **not** scored as confirming H5; it is reported as "state
dependence in the direction opposite to the project's target behaviour", which is a finding in its
own right and is exactly the outcome the freeze-to-heal literature would predict for an unmodulated
agent.

#### B1-B3 — companions, not verdicts

| ID | Measure | Definition | Role |
|---|---|---|---|
| **B1** | **Rest rate by starting-injury bin** | Fraction of steps on which the agent chose `Rest`, top versus bottom bin. | **Descriptive.** The dominant wound-driven behaviour in the existing record (+17.7 to +35.2 percentage points, 14 of 14 agents). It is what B0's denominator removes, so it is reported next to B0 so a reader can see how much freezing each arm does. It carries no threshold and no verdict. |
| **B2** | **Bush occupancy conditioned on acting** | Fraction of steps in a bush, counting only steps where the action was not `Rest`, same bin contrast. | **Descriptive only, demoted on 2026-09-07.** Retained for continuity with the fourteen-agent record and because it is nearly free to compute. **Its previous "flip toward zero or positive" threshold is withdrawn** and no hypothesis is scored on it. |
| **B3** | **Internal-state dependence of the threat response** | `B0(predator near) − B0(predator far)`, computed within each starting-injury bin, and the range of that difference across bins. Built on **B0**, i.e. on moving steps only. | **Secondary evidence for H5.** This is the interaction the modulator is supposed to produce: the response to an external cue changing with internal state. Scored on moving steps so that a frozen agent shrinks the sample rather than the effect. |

The original B3 — the same range computed on bush *occupancy* via
`scripts/analysis/context_dependence.py --state felt_pain` — is still run and still reported, but as
a **companion to B1**, under the standing rule that any change in it is discounted by whatever
change B1 shows in the same arm. It is not the primary evidence for anything.

#### The control band as measured — and what it says about the unmodulated agent

**This is the pre-registration's load-bearing step, and it has now been done.** `Δ_B0`, `B1`, `B2`
and `B3` were computed on the **five existing unmodulated `MC` seeds** (`rppo_cmp10m_mc_s42`
through `s46`), which share this grid's environment, agent configuration and budget with cell 1.
Their trajectory stores are on disk under `results/trajectories_nmnsite/` — 9.6 GB, 300,000 episodes
per run, `seed_base` 1,000,000, final checkpoints. The measurement gives two things: the **level**
of `Δ_B0` for an unmodulated agent in this exact world (the null H5-add is scored against) and its
**five-seed dispersion** (the only estimate this project has of how far a one-seed behavioural
number can move for reasons that have nothing to do with the modulator).

**Definition of the band**, unchanged: the min-to-max range of `Δ_B0` across the five seeds,
computed **per (window x predator condition) cell**. A grid arm counts as exceeding the control only
if its `Δ_B0` lies **above the top of that band**, in the predicted positive direction; §2.5 sets
the per-arm and count-across-targets rulings, and `T` / `W` there refer to this band's top and
width. Under rule C7, if the training path is confirmed unchanged the band is used for both level
and dispersion; if it is not, the band is used for **dispersion only**, cell 1 supplies the level,
and the band's width is carried over.

Two numbers accompany every arm and are never conflated: the **within-run** binomial 95% confidence
interval on `Δ_B0` (how precisely that one agent has been characterised) and the **five-seed control
band** (how much a second seed of the *same* configuration could move). A difference that clears the
first but not the second is not a result.

##### The measured band — on the OLD edges, and why it must be recomputed

**These are the values as measured on the superseded edges** (`0–25` versus `≥50`, after the empty
"exactly 0" bin forced a substitution on all five runs — see the binning discussion above). They are
recorded here because they show the **resolution** this design has to work with, which does not
change materially with the edges. **They are not the band the thresholds will be scored against**;
that band is recomputed on the re-registered edges `[25, 50, 75]` before any grid behavioural number
is read (§6.3).

| Reading | Five-seed range of `Δ_B0` | Band width `W` | Within-run 95% CI |
|---|---|---|---|
| First 25 steps, predator far (**the primary cell**) | −0.526 to +0.102 pp | **0.629 pp** | ±0.10 pp |
| Whole episode, predator conditions pooled | −0.248 to +0.010 pp | **0.258 pp** | ±0.10 pp |

Base rate of B0 itself across the five runs: **6.9% to 8.1%** of qualifying steps.

**The line to reckon with**: the within-run interval is **±0.10 pp** and the five-seed band is
**0.63 pp** — roughly **six times wider**. An arm reading `Δ_B0 = +0.3 pp` clears its own error bar
comfortably and still sits **inside** the null band. This is why every behavioural threshold in §2.5
is stated against the band and none against the interval, and why §2.5 says in the same breath that
**no single cell of this grid can confirm a behavioural hypothesis**.

**Pre-launch, this band is FINAL-checkpoint only.** The checkpoint rule below asks for two readings
per run — nearest-to-matched-budget and final — and only the **final** one is computable today,
because the matched budget is not defined until the grid runs establish a common environment-step
count. That matches this design's own sequencing; it is stated explicitly so that nobody later reads
the pre-launch band as if it were the matched-budget band.

##### A substantive finding, not a footnote: the unmodulated control does not show the target behaviour

Measuring the band produced a result about the **control** that changes what this grid is asking,
and it is recorded here as a finding rather than as calibration exhaust.

1. **The target behaviour is absent, and if anything leans the other way.** Starting an episode
   badly wounded changes the chance that a step out of the open lands in cover by between
   **−0.53 and +0.10 percentage points**, against a base rate near **7%**. **Three of the five seeds
   are negative.** The project's target behaviour — more cover-seeking when hurt — is not present in
   the plain agent at a magnitude this measurement can see.
2. **What the wounded agent does instead is stop moving, by a margin about thirty times larger.**
   B1, the rest rate in the top injury bin minus the bottom bin, moves **+18.2 to +20.9 percentage
   points** across the five seeds. The freeze-to-heal effect that B0 exists to exclude is roughly
   **30x** anything B0 itself moves. B0's design — scoring only steps on which the agent *chose to
   move* — is therefore not a refinement; it is the difference between measuring a decision and
   measuring a posture.
3. **The demoted B2 shows the artefact cleanly, on real data.** On seed 42, plain bush occupancy
   reads a **+2.83 pp per bin** trend with injury — which reads as "hides more when injured" —
   and excluding rest steps **flips it to −0.86**. One measurement, two opposite headlines,
   depending only on whether frozen steps are counted. This is the concrete instance of the artefact
   that `plan-reviewer` finding 3 warned about, and it retroactively justifies withdrawing B2's
   threshold.

**Consequence for how H5 is read**, stated before any grid data exists: **H5-add is not asking
whether the modulator amplifies an existing tendency. It is asking whether the modulator creates
one.** That is a harder question and a cleaner one — there is no baseline effect for a modulated arm
to inherit — and it makes an H5-add refutation (§2.5) correspondingly more informative: it would say
the mechanism fails to produce the behaviour at all, in the one place the project's thesis needs it.

#### Checkpoint rule

Every behavioural measure is computed at **two checkpoints per run, both reported**
(`plan-reviewer` finding 8):

1. the checkpoint whose logged environment-step count is **nearest the common matched budget** used
   for the primary survival reading (§4.1) — this is the comparison that is not confounded by the
   episode-budget artefact C8, under which arms differ in total experience by up to 3x; and
2. the **final** checkpoint of each run, for continuity with how every previous behavioural analysis
   in this project was collected.

Where the two disagree, the matched-budget reading carries the verdict and the disagreement is
reported.

**State as of 2026-09-07, stated so the pre-launch band is not over-read.** The five baseline seeds
have been collected at their **final** checkpoints only, and that is the only band that exists
today. Reading (1) — nearest-to-matched-budget — **is not computable until the grid runs define a
common environment-step budget**, because there is nothing yet to match to. Once that budget is
known, the five baselines are re-collected at their nearest checkpoints under the same spec
(§3.3); they save a checkpoint every 200,000 episodes, so a nearby one always exists. Until then,
every band quoted in this document is a **final-checkpoint** band and is labelled as one.

#### Collection parameters

Pre-registered so the collections are comparable: **300,000 episodes per checkpoint**, on
`basic/04` — the training world, not a purpose-built probe scene — with **`seed_base` = 1,000,000
shared by every run in the grid and every baseline seed**, which makes all arm-to-arm comparisons
**paired** on identical episode draws. `obs_precision: float32`; strict restore checking. The policy
is deterministic (argmax) at collection time, which is confound C11.

**These parameters are no longer prose — they are a spec file.** §3.3 names two collection specs
under `configs/trajectory_collection/`: one that reproduces the five-baseline collection already on
disk, and one that collects the 16 grid arms. **The `seed_base` value 1,000,000 is the load-bearing
one**: if a grid arm were collected under any other base, its episodes would be different worlds
from the baselines' and the comparison would degrade from **paired** to **unpaired** — silently,
with no error and no warning, because both collections would succeed. That is exactly the failure
the spec files exist to prevent.

(The first version of this design said "roughly 200 evaluation episodes per run"; that was simply
wrong and is corrected here and in §2.3.)

### 4.3 Diagnostic outcome — is the modulator doing anything at all?

Per-site gain and offset statistics, logged by the refactor (`modulator/gamma_<site>_{mean,std}`,
`modulator/beta_<site>_{mean,std}` for `uni`, `multi`, `rnn`, `actor`, `critic`), read as
trajectories over training, not as end-points.

- **Engagement**: does `gamma_<site>_std` leave zero? Every FiLM head is initialised as a no-op
  (gain 1, offset 0), so a standard deviation that never departs from zero means that site received
  no useful gradient and the arm is a null **for the mechanism**, distinct from a null for the site.
  This distinction is pre-registered in §5, item 2.
- **Stability**: does the gain drift monotonically or blow up? The new sites' gains are not clipped
  (only temperature and the legacy memory bias are), so an exploding gain is a real instability
  finding rather than a saturation artefact.
- **Secondary training diagnostics**: policy entropy, value loss, explained variance where
  computable, and gradient norm — read as trajectories, for the failure-mode rulings in §5.

### 4.4 What must be re-run with more seeds before anyone believes it

Pre-committed, so that the follow-up is scoped before the results create an incentive to widen it.

| Priority | Follow-up | Seeds | Why |
|---|---|---|---|
| 1 | The **best and worst single-site targets by the §2.3 bands**, plus the unmodulated control, at the winning input slice. | 5 each | Turns a ranking into a claim. Roughly 15 runs. |
| 1b | Any arm ruled **trap, unresolved** by §5 item 8 — a site that landed on a low plateau and therefore cannot be told apart, at one seed, from a site that does not work. | 2 more each | Prevents a good site being eliminated by one bad draw. This is the mitigation for the elimination asymmetry stated in §2.3; without it the grid's negatives are not trustworthy. |
| 2 | The **I-versus-X contrast** at the two write targets that survive priority 1, **and the I-versus-control contrast at the same targets** (H5-add is the load-bearing part of H5). | 5 each, so 20 runs | H5 is the project's mechanism hypothesis; it cannot rest on n = 1. |
| 3 | The **width-matched interoception control** (confound C3): `Extero Nociception` alone, 1 dim, against `Satiation` + `Interoceptive Nociception`, 2 dims, at the winning target. | 5 each | The only available way to separate "which information" from "how much information" using a name-keyed sensor selector. |
| 4 | Any arm that produced a **NaN or a value explosion** at a second seed (see §5, item 1). | 2 more | Distinguishes a bad run from an unstable site. |

**Not committed**: seeding all 15 modulated cells. That is 75 runs and roughly 900 GPU-hours to
answer questions most of which this screen will have already eliminated.

**An honest accounting of the total.** This grid is 16 runs and its committed follow-up is roughly
35 more, so the pre-registered programme is about **51 runs**. The staged alternative in §11 reaches
n = 3 on every H1-H4 cell in **28**. That comparison is the substance of the open question in §11;
it is recorded here so the follow-up cost is visible at the point the follow-up is scoped.

---

## 5. Failure-mode catalog (pre-decided)

Each ambiguous outcome is ruled **now**, before the data exists.

1. **Training instability — NaN, value explosion, loss divergence.** Refutes **the run**, not the
   hypothesis. Ruling: relaunch once at a different seed and record **both** attempts in the
   manifest. If it recurs at the second seed, it refutes **the site as configured** and is reported
   as "site unstable at these hyperparameters" — not as "this site does not help". A site that
   cannot be trained is a real result and should be written as one.

2. **The modulator never engages** — `gamma_<site>_std` stays at zero for the whole run. Ruling:
   this is a **null for the mechanism**, not for the site. Report as "the modulator did not engage
   at this site", and check whether the head received gradient at all (the refactor's check V5
   covers this at unit-test level; if a site is constructed but never read, that is a bug to route
   back to `developer`, not a scientific result).

3. **Gain saturation or explosion.** The new sites' gains are unclipped, so there is no ceiling to
   saturate against. A gain drifting to a large magnitude is therefore a **design finding** — the
   site is unstable under FiLM at this grouping — not a null result and not a measurement artefact.
   Report the trajectory.

4. **Insufficient horizon / right-censoring.** Ruling, adopting the prior study's discipline: **any
   arm whose survival is still improving by more than 5 steps over the final 20% of its episode
   budget is reported as right-censored and is excluded from every ranking claim.** Its number may
   be reported, labelled as a lower bound. This rule exists because the prior study's headline error
   was reading a *delay* as a *ceiling*: three arms that looked permanently broken at 1M episodes
   were merely ten times slower, and recovered fully given ten times the budget.

5. **Seed noise drowning the effect.** This is the *expected* outcome for most cells and is not a
   failure. Ruling: accept the null for those cells; do **not** add seeds to all 15. Escalate only
   the cells named in §4.4.

6. **Everything ties — all 15 modulated cells within 15 steps of the control** (the §2.3 no-evidence band). This is a
   substantive result, not a wasted experiment: it would say that under a uniform FiLM operator at
   this grouping and this budget, none of the four write sites buys survival in this world. Given
   that Paper 1 commits to one shared modulator at the sensory front-end, that outcome is a
   portfolio-level decision point. Ruling: escalate to `pi` with the behavioural measures of §4.2
   as the second line of evidence, since a modulator can produce context-dependent policy without
   producing extra survival — and for this project's thesis, the behaviour is what matters.

7. **An input arm fails to construct** because a sensor name does not resolve against the
   environment's observation breakdown. Ruling: this is a config error caught before launch by
   `env-config-reviewer` and by Part B's own loud-failure requirement, not an experimental outcome.
   It must never reach a training node.

8. **An arm lands in a low-performance trap and plateaus there — "trap, unresolved".** Added
   2026-09-07 on `plan-reviewer` finding 6. Item 4 above catches an arm that is *still climbing*; it
   does **not** catch an arm that fell into a bad basin early and then settled, because a settled
   trap passes the "still improving" test exactly as a genuine ceiling does. At one seed these two
   are not distinguishable, and the prior study shows the trap is real: five seeds of one
   configuration finished between 35.5 and 136.7 survival steps. **Ruling — an arm is marked "trap,
   unresolved" if any of the following holds**, and a marked arm is **not** scored as a resolved
   negative, is **not** eliminated, and goes to the §4.4 priority-1b follow-up:
   - it finishes more than 40 survival steps **below** the unmodulated control (a deficit larger than
     any this configuration family has shown between working variants, and therefore more likely a
     basin than a site effect); **or**
   - its survival curve shows a plateau reached before 30% of the budget with less than 5 steps of
     subsequent movement — early arrival at a low ceiling, which is what a trap looks like; **or**
   - its modulator engagement diagnostics (§4.3) show the gain spread collapsing to zero after
     having left zero — the site learned something and then switched itself off.

   **Why this is deliberately conservative**: it is better to hand back "we could not resolve this
   cell" than to record "this site does not work" on the strength of one draw, because the second
   statement will be cited and the first will be re-run. The cost is that this grid will eliminate
   fewer cells than a 16-cell screen appears to promise — which is the argument §11 makes for the
   staged alternative.

9. **Every arm's `Δ_B0` sits inside the control band.** Pre-decided 2026-09-07, once the band was
   measured and its width was known. Ruling: this is the **modal expected outcome**, not a failure,
   and it is scored exactly as §2.5 says — H5-add **refuted** at 1 or 0 of 5 targets clearing the
   band, **unresolved** at 2. It must be written up with its own limit attached: the grid can only
   see `Δ_B0` effects larger than roughly one band width (about 0.6 percentage points in the primary
   window, against a ~7% base rate — a 9% relative change), so "refuted" means *no effect larger
   than that*, never *no effect*. It may **not** be written up as "the modulator does nothing"
   without that clause. Given the pre-launch finding that the unmodulated control shows **no**
   cover-seeking state-dependence to begin with (§4.2), this outcome would say the modulator fails
   to **create** the target behaviour — which is the more consequential reading and the one to lead
   with.

---

## 6. Metrics and tooling requested

### 6.1 Metrics not currently logged

| Subfield | Content |
|---|---|
| **Metric** | **Per-step modulator signals in the trajectory store** — for each enabled site, the mean and standard deviation of the gain and of the offset across the 128 modulated neurons, recorded per step alongside the existing state columns (four floats per site per step). Unit: dimensionless gain / offset. |
| **Why now** | Hypothesis H5 and behavioural measure B3 ask whether the modulator's gain **tracks the internal state**. The training-time logs give only batch-wide means and standard deviations, which show *that* the gain varies but never *what it is a function of*. The trajectory store is where the state columns live, and it currently discards `mod_info` entirely — the model returns it and the collector drops it. Without this, the representational half of the project's 4x3 framework cannot be read for any run in this grid. |
| **Where it'd live** | `scripts/eval/traj_collect/traj_scan.py` (carry `mod_info` through the scan alongside `action`) and `scripts/eval/traj_collect/collect_trajectories.py` (add the columns to the parquet schema). |
| **Cost** | **Moderate.** The reduced form above is 4 floats per site per step — with four sites that is 16 extra float columns, comparable to the existing per-step state columns. A full per-neuron dump (128 values per site per step) would be **expensive** and is explicitly *not* what is requested; if a "which neurons" analysis is later needed it should be a separate, opt-in flag. |

### 6.2 Analysis tooling requested (not new metrics)

Neither item below is a training-time metric, so **neither gates the launch of the grid**. Both are
small additions to existing analysis scripts. Item (a) has since been **built and run**; item (b) is
still outstanding and still non-blocking.

**(a) The bush-entry rate (B0) — DONE, 2026-09-07.** The measure was built into
`scripts/analysis/context_dependence.py` and run against all five existing unmodulated `MC` seeds.
Every column it needs already existed in the trajectory store (`agent_in_bush`, `action`, `rested`,
`agent_row`, `agent_col`, `animal_row`, `animal_col`, `animal_active`, plus the episode table's
starting draws), so it was arithmetic over data on disk rather than new logging. The measured
control band, the resolution it implies, and the substantive finding it produced about the
unmodulated agent are in §4.2. **The exercise also found a flaw in this pre-registration** — the
structurally empty "exactly 0" injury bin — which is why §4.2's binning was re-registered and why
§6.3 below requests one recomputation.

**Ordering rule, stated so nobody has to guess under time pressure.** It now applies to the §6.3
recomputation rather than to the original build, and it is unchanged in substance: the launch does
**not** wait on it. The hard requirement is that the band on the re-registered edges exists **before
any behavioural number from a grid run is read**. That is not a weakening of the pre-registration,
and the reason is worth stating plainly — the band is measured on five runs that finished on
2026-09-05, which no grid run can influence; the bin edges are fixed in §4.2 by an argument that
depends only on the environment's sampling distribution, not on any measured value; and the rule for
turning the five seeds into a band (min-to-max of `Δ_B0`, per window and predator condition,
exceeded only from above in the positive direction) is fixed in §4.2 and §2.5 above. What would
break pre-registration is choosing the edges or the band *after* seeing the grid's behaviour, and
that is forbidden either way.

**(b) The critic-value pre-check for H3 (§2.5).** Plot the distribution of critic values conditioned
on injury, on the existing `MC` checkpoints. The value output is computed inside
`scripts/eval/eval_rollout.py` (`get_action_and_value_nnx`) and discarded, and the trajectory store
deliberately does not record it, so this needs a short script that replays stored observation
sequences through a loaded checkpoint's critic — recurrent, so per-episode sequential replay. Its
result **does not gate the launch and does not move any H3 threshold** (§2.5); it changes only
whether an H3 null reads as expected or as surprising.

### 6.3 Requested: recompute the control band on the re-registered bin edges

**Handed back as a request, not performed here** — this agent designs and configures; it does not
run the analysis.

**What.** Recompute the five-seed control band for `Δ_B0`, and the companion measures `B1`, `B2` and
`B3`, on the **re-registered injury bin edges `[25, 50, 75]`** (four bins 0–25 / 25–50 / 50–75 /
75–100; `Δ_B0` = 75–100 bin minus 0–25 bin). The edges live as the module constant `INJ_EDGES` in
`scripts/analysis/context_dependence.py`, currently `[1e-9, 25.0, 50.0]`, so this is a one-constant
change plus a rerun — **but it is a change under `scripts/`, which is outside this agent's write
surface**, so it is routed rather than made. Whether the edges become the new constant or a
command-line option is the implementer's call; if it becomes an option, the value used must be
recorded in the output alongside the numbers.

**Why it is cheap.** No re-collection is needed. All five trajectory stores are already on disk at
`results/trajectories_nmnsite/` — 9.6 GB, 300,000 episodes per run, `seed_base` 1,000,000, final
checkpoints — and the recomputation is a re-read of data that already exists.

**What is needed back**, per (window x predator condition) cell, for both windows (first 25 steps,
whole episode) and all three predator conditions (near, far, pooled):

- the five per-seed values of `Δ_B0`, and the band `T` (top) and `W` (width = max − min) derived
  from them — these are the constants §2.5's thresholds are written against;
- the within-run 95% confidence interval on `Δ_B0` for each seed, so the seed-band-versus-interval
  ratio can be restated on the new edges;
- the base rate of `B0` itself, and the realised denominator count in each of the four bins, so the
  2,000-step minimum-count rule can be confirmed not to fire;
- `B1` (rest rate, top bin minus bottom bin) and the `B2` with/without-rest pair, on the new edges,
  for the descriptive record.

**Label everything final-checkpoint.** The nearest-to-matched-budget reading is not computable until
the grid runs define a common environment-step budget (§4.2, checkpoint rule).

**When.** Before any behavioural number from a grid run is read. It does **not** gate the launch —
the band is measured on five runs that finished on 2026-09-05 and cannot be influenced by anything
the grid does, and the rule for turning them into a band is fixed in §4.2 and §2.5 above. What would
break pre-registration is choosing the band *after* seeing the grid's behaviour, and that is
forbidden either way.

### 6.4 What is already covered and needs nothing

- The per-site gain and offset training-time series (§4.3) are in the site refactor's own File
  Changes section and will exist when Part A lands.
- The return-scale metrics requested by the return-mode study (`returns/std`, `targets/std`,
  `advantages/std_preclip`, `value/explained_variance`) would be useful here too but are **not
  requested by this design** — nothing in §2.5 or §4 is bottlenecked on them.


---

## 7. Results

*To be filled after training. Left blank deliberately.*

## 8. Analysis

*To be filled after training. Left blank deliberately.*

## 9. Conclusions

*To be filled after training. Left blank deliberately.*

---

## 10. Response to the plan review (2026-09-07)

**What this section is.** `plan-reviewer` returned a verdict of **NOT READY** on the first version
of this design. This section records, finding by finding, what changed in the document, what did not
change and why, and the two places where I did something other than what the review asked for. It is
here so that a future reader can see the design's failure modes as well as its claims — the review
caught a real error that would otherwise have produced a wrong claim about the agent's behaviour.

**The one that mattered.** The headline behavioural test — "does the wounded agent seek cover?" —
was measuring *where the agent was standing* rather than *what it decided to do*, and was scoring
that against a reference number borrowed from a differently-defined quantity. Since a wounded agent
in this world mostly **stops moving**, that measure would have reported "hides more" or "hides less"
for an agent that had merely frozen more or less often, in a bush or out of one by luck. The
replacement asks only about steps on which the agent **chose to move**, and counts how often that
step went into cover. Freezing now removes a step from the measurement instead of contributing to
it. Full definition in §4.2.

| # | Finding | Disposition | Where |
|---|---|---|---|
| 1 | Part B (the code letting the modulator read a subset of the senses) is not in the tree | **Not mine to clear.** The launching agent owns it. The design already blocks on it as dependency D-B. | §2.7 |
| 2 | The working tree is uncommitted, so no run is tied to a commit | **Not mine to clear**, but a `Code SHA` column was added to the Launch Manifest so the answer has somewhere to live, with the all-16-identical / `git_dirty: false` rule stated. | §3 |
| 3 | 🔴 The behavioural measures restate the freeze-to-heal trap rather than escape it | **Accepted in full.** §4.2 rewritten around a motion-based headline measure (bush-entry rate, B0); the old B2 threshold **withdrawn** and B1/B2 demoted to descriptives; B3 rebuilt on moving steps only; the null band is now **measured before launch** on the five existing unmodulated seeds instead of borrowed. | §4.2 |
| 4 | `lr_critic` requirements 2 and 5 contradict each other | **Accepted.** `0.0005` in all 16 files including the control; requirement 5 reworded to "identical except `lr_critic`", with the reason recorded. | §2.4 C5, §3.1 |
| 5 | The 10-step floor was justified by wrong arithmetic; best-slice scoring is a max-of-three | **Accepted.** Bands raised to 15 / 30 with the corrected arithmetic stated openly; every target-vs-control hypothesis is now scored **at the ALL slice** as primary, best-slice as a labelled secondary. | §2.3, §2.5 |
| 6 | Elimination is ungated; a good site can be dropped on one bad draw | **Accepted in part, by instruction.** The **"trap, unresolved"** rule is added (§5 item 8) together with a follow-up slot (§4.4 priority 1b) and an explicit statement of the asymmetry (§2.3). The **design's size and shape are unchanged** — that was the user's decision and it is not being revised while they are unavailable. The reviewer's staged alternative is recorded in §11 for the user to rule on. | §2.3, §4.4, §5, §11 |
| 7 | H5 (I beats X) is close to true by construction | **Accepted.** H5 split into **H5-add** (I versus the unmodulated control — the load-bearing claim, since the main network sees all 27 dimensions in every arm) and **H5-spec** (I versus X — the specificity check, which can never confirm on its own). | §1.4, §2.5 |
| 8 | No checkpoint rule for the behavioural measures | **Accepted.** Nearest-to-matched-budget **and** final, both reported, matched-budget carries the verdict. | §4.2 |
| 9 | The H3 citations are accurate but the conditioner in both papers is privileged information | **Accepted with one deviation** (see below). The third caveat is now carried next to the prediction, and the prediction is explicitly held *weakly*. | §2.5 |
| 10 | `percept_bias_init: 3.0` is inert under FiLM | **Accepted.** Listed as inert, removed from the load-bearing set. | §2.2 |
| 11 | The eval-boundary compatibility shim deviates from refactor decision D16 | **Not mine.** `senior-developer`'s adherence check. | — |
| A1 | "Does the unmodulated path still reproduce the known result?" is an untested assumption | **Accepted.** Added as the pre-registered **C7 gate**, scored before any hypothesis, with a void-pending-diagnosis ruling if it fails. | §2.4 C7, §4.1 |

### 10.1 Two places where I did not do exactly what was asked, and why

**(a) The critic-value pre-check for H3 is requested, but it does not gate the launch.** The review
calls it "free". It is not quite free: the critic's value output is computed and discarded in the
eval path, and the trajectory store deliberately does not record it, so the check needs a short new
script that replays stored observations through a loaded checkpoint's recurrent critic. That is
small, but it is development work, and gating a 16-run launch on it would trade a real cost for a
diagnostic that **changes no threshold in this design** — its only effect is on whether an H3 null
reads as expected or as surprising. It is therefore requested in §6.2(b), the ruling for both of its
outcomes is pre-registered in §2.5, and the launch does not wait on it. The B0 tooling in §6.2(a) is
different and **does** have to exist before any behavioural number is read, because the null band
depends on it.

**(b) The injury breakdown uses fixed bin edges, not quartiles — and that answer was half wrong.**
The instruction was to break B0 down by starting-injury **quartile**. The original response refused,
on the ground that quartiles are defined by the sample, so the boundary between "lightly wounded"
and "badly wounded" would sit at a different injury value in each of the sixteen arms and the arms
would stop being comparable on the axis the whole measure is about. That reasoning is correct, but
the substitute chosen — the edges already implemented in `scripts/analysis/context_dependence.py`
(*exactly 0* / *0–25* / *25–50* / *50+*) — was not, and building the measure exposed why: the
environment draws starting injury from a **continuous** uniform distribution, so the *exactly 0* bin
is **structurally empty** and the realised contrast was quietly *0–25* versus *≥50* on all five
baseline runs.

**Resolved 2026-09-07, and the two requirements turn out not to conflict at all.** Because the draw
is uniform on 0–100, the **population** quartile boundaries are the known constants **25 / 50 / 75**
— fixed numbers, not sample statistics. Adopting them gives quartiles (four equal-probability bins,
as the review asked) *and* identical edges in every arm (as this design required), simultaneously.
The re-registered edges are `[25, 50, 75]` and `Δ_B0` is the 75–100 bin minus the 0–25 bin. Full
reasoning, including the cost — loss of numeric comparability with the fourteen-agent reference
record — is in §4.2. The recomputation of the control band on the new edges is requested in §6.3.

---

## 11. Open for the user — a smaller, replicated alternative to this grid

**Read this if you are deciding what to spend the next 16 GPU slots on.** It is a recommendation,
not a change. This design proceeds as approved unless you say otherwise.

**The problem with the approved design, in one paragraph.** Sixteen cells at one seed each gives a
number for every combination and a confidence interval for none. Positive results are protected — a
gain above 30 survival steps has to be re-run at five seeds before anyone may call it an effect —
but *negative* results are not. A site that would work fine, handed one unlucky random seed, gets
recorded as "does not help" and dropped. That is not hypothetical in this project: five runs of one
identical configuration in the immediately preceding study finished anywhere between 35.5 and 136.7
survival steps. The "trap, unresolved" rule added at §5 item 8 blunts this, but it blunts it by
refusing to conclude — so the grid will return fewer clean answers than sixteen cells suggests.

**The alternative, from `plan-reviewer`, using the same number of slots.**

| Stage | Runs | What it answers |
|---|---|---|
| **Stage 1** | 5 write targets x 3 seeds, all at the full-input (ALL) slice, **= 15 runs**, plus 1 unmodulated control run for the replication gate = **16** | Every "does the write site matter, and which one" question (H1-H4) at **three seeds each**, with the five existing unmodulated runs as the control's own replicated baseline. |
| **Stage 2** | The 2 targets that survive stage 1 x (I, X) x 3 seeds = **12 runs** | The interoception question (H5, H6) at three seeds, on the sites that turned out to be worth asking about. |
| **Total** | **28 runs** | Every cell it reports carries n = 3. |

Against this design's **16 runs now plus roughly 35 committed follow-up runs (§4.4) = about 51**,
for cells that carry n = 1 until the follow-up lands.

**What the staged version buys**: replication on every survival comparison, so eliminations become
trustworthy and the untested assumption that modulated arms are as seed-tight as unmodulated ones
gets measured rather than assumed. **What it costs**: the interoception question — the one this
project's thesis rests on — is deferred by one wave (roughly 15 hours), and a write site that only
works with a *restricted* input would never be seen, because stage 1 tests every site at the full
input slice only. That risk is real but the prior for it is low.

**If you want the staged design but the grid has already launched, nothing is wasted.** The grid's
six ALL-slice runs (the control plus one seed at each of the five targets) are exactly stage 1 at
seed 42. Completing stage 1 then costs **10 further runs** (two more seeds at each of the five
targets), and the grid's ten I / X runs become a one-seed preview of stage 2 rather than a dead end.
The two designs are nested, not alternatives, which is why launching the grid does not foreclose
the choice.

**Designer's recommendation**: the staged design is the better use of the slots, for the reason the
reviewer gives — a replicated negative is worth more than an unreplicated ranking, and this grid's
main product is negatives. **The decision is yours**, and the approved 16-cell grid is what this
document specifies until you rule otherwise.

---

## Appendix

### A. Where each fixed number in this design came from

| Number used | Source | Verified how |
|---|---|---|
| 27-dim observation, 7 sensors, per-sensor widths | live run banner + `src/environment/sensor.py::get_observation_breakdown` | Arithmetic re-derived from `configs/environment/default.yaml`: olfactory grid range 0 x 5 resource properties = 5; collision sensor range 1 -> 5 cells; proprioception = 6 actions; visual range 0 x visual vector size 8 = 8. Sums to 27. |
| `injury_observable: false`, `nutrition_observable: false`, `injury_smoothing_duration: 3` | `configs/environment/default.yaml:256,257,178` | Read directly; `basic/04` inherits through `basic/03` and does not override them. |
| 5-seed survival dispersion for `MC` (1.89 sd, 4.5-step range) | [[return_mode_cmp_10M]] §4.1 | Same environment, same agent config, same 10M-episode budget. |
| Dispersion at matched experience (2.7 / 9.4 / 26 / 32 / 83) | [[return_mode_cmp_10M]] §4.2 | Per-seed table at 349 M environment steps. |
| Order-of-magnitude experience penalty for split-scale return modes (6x-14x) | [[return_mode_cmp_10M]] §1, §4.5 | Measured escape times and matched-budget survival. |
| Rest-rate rise with injury (+17.7 to +35.2 points, 14/14 agents) | [[INJURY_HIDING_SIGN_RECONCILIATION]] §4 Finding 1 | Recomputed there over the 14 sensor-ladder arms. |
| Acting-conditioned cover use falls with injury (−0.77 to −9.49 points, 14/14) | [[INJURY_HIDING_SIGN_RECONCILIATION]] §4 Finding 2 | Same source; both predator-present and predator-absent conditions. |
| Window-dependent sign flip (13/14 -> 2/14 -> 1/14) | [[INJURY_HIDING_SIGN_RECONCILIATION]] §3 | Three windows over the same runs. |
| Value-head FiLM cut error 42-57% | [[MODULATION_SITE_REFACTOR]] Analysis §F, citing Marquis & Farhood | Corpus's only controlled actor-vs-critic PPO ablation. |
| PAPL's value-modulation precondition is met | [[MODULATION_SITE_REFACTOR]] Analysis §F | Traced to `src/environment/core.py:49-53` and `:728-731`; note the reader's footgun there — `calculate_drive` is the reward's drive, while `drive_hunger` / `drive_injury` are logging-only. |
| Wall clock ~10.8 h for 10M episodes | run directory `results/JAX_RecurrentPPO/20260904-173804_rppo_cmp10m_mc_s42` | Directory mtimes, first to last write. |
| No tag collision on `rppo_nmnsite_` | `results/JAX_RecurrentPPO/` | 387 run directories scanned; zero matches. |
| Movement actions are 0-3; `Rest` is 4, `Eat` is 5 | `src/environment/core.py:580-581,660-661` | `rested = (action == 4)` when the rest action is enabled; `eat_action_idx = 5` under the same condition. **Only the `Rest` index has a witness in the store.** The guard actually applied is that no row may carry a movement action and the `rested` column at the same time — verified true on all five baseline runs. `Eat` has **no** stored column that identifies it (`ate_food` is `False` when `Eat` is chosen on an empty cell), so its index 5 is taken on trust from the source. See §4.2. |
| Predator-near threshold: Chebyshev distance 2 | `scripts/analysis/context_dependence.py:53` (`NEAR_D = 2`) | Reused unchanged so B0 and the existing occupancy measures share one definition of "near". |
| Starting-injury bin edges **25 / 50 / 75** (bins 0-25 / 25-50 / 50-75 / 75-100) | **Re-registered 2026-09-07.** The population quartiles of the environment's own starting-injury draw. | The draw is `jax.random.uniform(minval=0, maxval=100)` (`src/environment/core.py:1103-1107`) with bounds from `basic/03-random_init_10x10.yaml:118-119`, inherited by `basic/04`. A uniform draw's quartile boundaries are the constants 25 / 50 / 75, so these edges are fixed *and* equal-probability. Supersedes `INJ_EDGES = [1e-9, 25, 50]` at `scripts/analysis/context_dependence.py:85`, whose first bin ("exactly 0") is structurally empty. See §4.2 and §10.1(b). |
| Starting injury is continuous on (0, 100); zero exact zeros in 15,000 episodes; minimum observed **0.0058** | Sampled from the environment during the 2026-09-07 B0 build | Consistent with a uniform draw on 0-100: the expected minimum of 15,000 such draws is about 0.0067. This is why the "exactly 0" bin cannot be populated. |
| Five-seed control band for `Δ_B0`: **-0.526 to +0.102 pp** (first 25 steps, predator far) and **-0.248 to +0.010 pp** (whole episode, pooled); base rate 6.9-8.1%; within-run 95% CI **+/-0.10 pp** | Measured 2026-09-07 on the five stores at `results/trajectories_nmnsite/` | **On the SUPERSEDED edges**, final checkpoints only. Recorded for resolution, not as the scoring band; §6.3 requests the recomputation on the re-registered edges. The seed band is about **6x** the within-run interval, which is why every behavioural threshold in §2.5 is stated against the band. |
| `B1` (rest rate, top minus bottom injury bin): **+18.2 to +20.9 pp** across the five baselines; `B2` on seed 42 flips from **+2.83** to **-0.86 pp/bin** when rest steps are excluded | Same measurement | The freeze-to-heal effect is roughly **30x** anything `Δ_B0` moves, and the `B2` flip is the artefact `plan-reviewer` finding 3 predicted, shown on real data. See §4.2. |
| `env_fp` differs across the five baseline stores and is **not** an environment difference | The five `_manifest.json` files (a4cbd376c9 / 0c7260c592 / 2770336d22 / 9e8a6ca5e1 / eb11489e2c) | `env_fp` hashes the whole resolved config including `seed`, `tag` and the wandb fields, so identical worlds fingerprint differently. Pairing was verified **empirically** instead: over 5,000 shared episode seeds the animal draws, observation draws, starting injury and spawn cell are bit-identical across all five stores. Every grid arm will likewise carry a distinct `env_fp`. See §3. |
| Early window = first 25 steps | `scripts/analysis/context_dependence.py:54` (`EARLY = 25`) | Matches the window of the reference record. |
| 300,000 episodes per trajectory collection | `results/trajectories_lad/*/*/*/_manifest.json` (`n_episodes: 300000`) | The sensor-ladder study's own collections, on disk, at ~2.0 GB each. Supersedes the earlier "roughly 200 episodes" figure in §2.3, which was wrong. |
| A checkpoint exists near any budget | `results/JAX_RecurrentPPO/20260904-173804_rppo_cmp10m_mc_s42/models/` | 52 checkpoints, one per 200,000 episodes, retained for each of the five `MC` seeds. |
| The store carries every column B0 needs | [[TRAJECTORY_STORE_SCHEMA]] §3.1-3.2 | `agent_in_bush`, `action`, `rested`, `agent_row`/`agent_col`, `animal_row`/`animal_col`, `animal_active`; row `t` holds state at `t` and the action that arrived at `t`. |
| The critic's value output is not stored | `scripts/eval/eval_rollout.py:317,1210` (`_value` discarded); schema's "deliberately NOT recorded" list | Why the H3 pre-check needs a script rather than a query. |

### B. Deviation from the standard template

This document reorders [training_analysis.md](../../../TEMPLATES/training_analysis.md) so that
everything pre-registered (§1, §2, §4, §5, §6) sits ahead of everything post-hoc (§7, §8, §9), and
the Analysis Plan appears **before** Results rather than after. Two sections beyond the template
follow the results placeholders: §10, the point-by-point answer to the plan review, and §11, the
alternative design recorded for the user to rule on. §3 keeps its number and its column
semantics unchanged, because `training-runner` and `experiment-analyzer` both key on
`## 3. Launch Manifest`.

### C. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-09-07 | Initial pre-registered design. Configs deliberately not produced — blocked on dependencies D-A, D-B, D-C (§2.7). | `experiment-designer` |
| 2026-09-07 | **Second amendment — corrections found by building the measure, before any run launched.** (1) **Injury bins re-registered** from `[1e-9, 25, 50]` (bins *exactly 0* / 0-25 / 25-50 / 50+) to **`[25, 50, 75]`** (bins 0-25 / 25-50 / 50-75 / 75-100): starting injury is drawn continuously on 0-100, so the "exactly 0" bin is structurally empty, the substitution rule fired on all five baselines, and every realised `Δ_B0` was a narrower contrast than the doc implied. The new edges are the population quartiles of a uniform draw, so they are fixed *and* equal-probability — which also resolves §10.1(b). (2) **Every behavioural threshold in §2.5 restated against the five-seed band, none against the within-run interval**, with the measured 6x gap stated (band 0.63 pp vs interval +/-0.10 pp), per-arm rulings tabulated, count-across-targets rules given explicit null probabilities, and the unresolvability of any single cell at n = 1 stated in the same place as the thresholds. (3) **A substantive pre-launch finding recorded** (§1.1, §4.2): in the unmodulated control the project's target behaviour is absent and three of five seeds lean negative, while the freeze-to-heal effect is ~30x larger — so H5-add asks whether the modulator *creates* the behaviour, not whether it amplifies it. (4) **Two trajectory-collection specs committed** (§3.3) at `seed_base` 1,000,000, replacing an ad hoc script. (5) `env_fp` recorded as **not** a same-world check (§3); the `Eat`-index verification limit recorded (§4.2); the pre-launch band labelled **final-checkpoint only** (§4.2). (6) Recomputation of the band on the new edges requested in §6.3. **Design size, shape, and all survival-side thresholds unchanged.** | `experiment-designer` |
| 2026-09-07 | **Amended in answer to `plan-reviewer`'s NOT READY verdict, before any run launched.** §4.2 rewritten around a motion-based headline measure (B0, bush-entry rate) with its null band measured pre-launch on the five existing unmodulated seeds; the old B2 threshold withdrawn and B1/B2 demoted to descriptives; B3 rebuilt on moving steps. Resolution bands raised from 10/25 to 15/30 with the arithmetic corrected. Target-vs-control hypotheses now scored at the ALL slice as primary. H5 split into H5-add (versus the control) and H5-spec (versus X). "Trap, unresolved" failure mode added (§5 item 8) with a follow-up slot (§4.4 priority 1b). C7 replication gate added (§4.1). `lr_critic` contradiction resolved (§3.1). Privileged-information caveat added to H3 plus a non-blocking critic-value pre-check. `percept_bias_init` marked inert. `Code SHA` column added to the manifest. §10 (response to the review) and §11 (the staged alternative, open for the user) added. **Design size and shape unchanged.** | `experiment-designer` |

---

## Feedback from plan-reviewer (2026-09-07)

**Verdict: NOT READY** — two launch-blocking findings (Part B not in the tree, so 10 of 16 cells cannot be configured and a staggered launch would leave the first 6 un-evaluable; the working tree is uncommitted, so no run can be tied to a SHA) and one analysis-plan Critical (the B2 headline threshold cites a share-of-all-steps band for a fraction-of-acting-steps measure, and both B2 and B3 remain exposed to the freeze-to-heal artefact; a bush-*entry* rate is the motion measure that escapes it). Survival-side noise floor, sensor layout, H3 citations and the withdrawal of Fix 1 all verified. Full table, assumptions and exit condition: [`docs/reviews/plan_nmn_input_site_grid.md`](../../../reviews/plan_nmn_input_site_grid.md).

— Reviewed by: plan-reviewer

### Response from experiment-designer (2026-09-07)

Findings 3, 4, 5, 7, 8, 9 and 10, and assumption A1, are answered in the document; finding 6 is
answered in part, by instruction, with the design's size and shape left as the user approved it and
the staged alternative recorded in §11 for the user to rule on. Findings 1, 2 and 11 belong to the
launching agent and to `senior-developer`. Two deviations from what the review asked for are stated
with reasons in §10.1: the critic-value pre-check is requested but does **not** gate the launch, and
the injury breakdown uses fixed bin edges rather than sample-defined quartiles. The point-by-point
disposition table is §10.

— `experiment-designer`
