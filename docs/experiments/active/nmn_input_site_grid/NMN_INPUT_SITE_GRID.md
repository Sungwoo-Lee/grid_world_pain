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

> **Status**: PRE-REGISTERED — design only. **Configs not yet written**; they depend on two
> config keys that do not exist in the code at the time of writing (see §2.7 Dependencies).
> **Date**: 2026-09-07
> **Author**: `experiment-designer`
> **Mode**: fully pre-registered. Every prediction, threshold and failure-mode ruling in §2.5,
> §4 and §5 is fixed **before** any run launches and may not be adjusted afterwards.
> **Related**: [[MODULATION_SITE_REFACTOR]] (the implementation this design consumes) ·
> [[return_mode_cmp_10M]] (source of the noise floor and of the return-mode choice) ·
> [[INJURY_HIDING_SIGN_RECONCILIATION]] (source of the behavioural-measurement rules) ·
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
> **H5** (*the modulator must read the body*): agents whose modulator reads only the interoceptive
> channels (**I**) show more internal-state-dependent behaviour than agents whose modulator reads
> only the exteroceptive channels (**X**), at the same write target.
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
unmodulated control, on survival steps at matched environment experience:

| Difference vs. control | Ruling | Justification |
|---|---|---|
| Below 10 steps in absolute value | **No evidence.** Not reported as an effect in any direction. | About 2x the largest observed 5-seed range for this configuration (9.4 steps at matched budget). |
| 10 to 25 steps | **Flagged, unresolved.** Reported as a candidate only; may not be described as an effect. | Above the observed range but not above it with margin for the untested assumption that modulated arms are as tight as unmodulated ones. |
| Above 25 steps | **Candidate real effect.** Promoted to a 5-seed confirmation before any claim is made. | Roughly 2.5x the observed range, and 13x the `MC` standard deviation. |
| Any arm judged fragile by the §5 criteria | **Unresolved regardless of margin.** | Lesson 2 above. |

**Nothing in this grid is confirmatory.** All 15 modulated cells are n = 1, so all 105 pairwise
contrasts among them are one draw against one draw. The grid's job is to **rank and to eliminate**,
producing a short list for a properly seeded follow-up — not to establish anything.

**One exception, and it is about precision, not generalisability.** The behavioural measures of
§4.2 are computed over roughly 200 evaluation episodes per run, so their *within-run* confidence
intervals can be narrow even at one seed. That narrowness measures how precisely each agent's
behaviour has been characterised; it says nothing about whether a second seed of the same
configuration would behave the same way. Both numbers are reported, and the second question is
answered only by the follow-up.

### 2.4 Confounds and limitations

| # | Confound | Severity | Assessment / mitigation |
|---|---|---|---|
| **C1** | **Single seed per cell.** | **Critical** | Not mitigated — it is the design. §2.3 states the resolution bands; §4.4 names the confirmation runs. Every verdict in §7-§9 must carry the n = 1 qualifier. |
| **C2** | **ALL and X differ by 8 dimensions, not 2** (both interoceptive channels *plus* six proprioception channels), because proprioception is deliberately in neither restricted slice. | **Critical for the ALL-vs-X reading** | Not mitigable within this design; it follows from the user's recorded decision. **The pre-registered rule: no ALL-vs-X difference is ever attributed to interoception.** The interoceptive contrast is I vs X, and only that one. |
| **C3** | **Input width is not matched across the input factor** (27 / 2 / 19). Any I-vs-X difference confounds *which* information the modulator reads with *how much*, and with the size of its input layer. | **High for H5** | Not mitigable with a name-keyed sensor selector. Stated in every H5 verdict. A width-matched follow-up is available and named in §4.4: `Extero Nociception` alone (1 dim) against `Satiation` + `Interoceptive Nociception` (2 dims). |
| **C4** | **Pinning `rnn_mechanism: activation` breaks continuity with the historical neuromodulated record**, every run of which used the legacy gate-bias operator on the memory site. | Medium | Deliberate. Uniformity across sites is what makes Factor 2 a single factor; continuity with a record that confounds site with mechanism is worth less. The trade is recorded so a future reader does not read cell `t3rnn` as a replication of past memory-site runs. |
| **C5** | **Learning rates.** Today one optimiser trains the shared trunk, both heads and the modulator at `lr_actor`, and the advertised `lr_critic` is read by nobody. If [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]] lands first, `lr_critic` and a new `lr_modulator` become live. | **Critical if unhandled** | **Pre-launch requirement**: all 16 configs must set `lr_actor = lr_critic = lr_modulator = 0.0005`. If they do not, the value-head target (5) and the all-four target (16) train their FiLM heads at a different effective rate from the other targets, and Factor 2 confounds with learning rate. Verified by `env-config-reviewer` before launch. |
| **C6** | **The meaning of `return_mode: MC` may change under the pending fix.** That fix's Part 1 proposes replacing the MC branch with "raw critic target, normalised advantage" — which is exactly the `MC_FIXED` mode that the later 25-run study measured as roughly an order of magnitude slower and 70 steps worse at matched experience. | **Critical, pre-launch blocker** | The two documents were written four days apart and have not been reconciled. **This grid must not launch until it is established which semantics `return_mode: MC` will carry.** If the fix lands as written, every arm here inherits a 10x experience penalty and the 10M-episode budget becomes far too small. Escalated to the user in the handoff; see §2.7. |
| **C7** | **Continuity of cell 1 with the five existing `MC` seeds** depends on no change to the training path between commit `788e5983` (those runs) and launch. | Medium | The site refactor promises the unmodulated path is bit-identical (its checks V1/C1). The pending fix does **not** — it changes loss values by design. **Pre-registered rule**: the five existing seeds are used for **dispersion** (how much seeds vary) and never for **level** (what the baseline is) unless the training path is confirmed unchanged. |
| **C8** | **Budget is in episodes, not environment steps, and better agents run longer episodes.** In the prior study total experience differed 3.3x across arms at a fixed episode budget. | **Critical** | Handled by design: the primary comparison is made at **matched environment steps** (the trainer logs `timesteps` directly), with the episode-axis number reported alongside. See §4.1. |
| **C9** | **Right-censoring.** Some arms may not have plateaued at 10M episodes. | High | Pre-registered exclusion rule in §5, item 4. |
| **C10** | **Checkpoints are not interchangeable across input arms.** Each slice changes the modulator GRU's input width, so the parameter trees differ. | Low — stated, not a defect | These are 16 fully independent trainings. No warm-starting, no shared initialisation beyond the common seed, no cross-arm checkpoint restore. Any attempt to restore across arms will fail loudly (the restore-completeness assertion), which is the desired behaviour. |
| **C11** | **The offline behavioural measures use a deterministic (argmax) policy** while the training-log survival series reflects the stochastic policy. They measure different objects. | Medium | Both reported side by side, never differenced. Same discipline as the prior study's C8. |
| **C12** | **`percept_add_bias_init` is read with a fallback default** (`recurrent_ppo_network.py:361` uses `.get(..., 0.0)`), contrary to the project's no-fallback rule. | Low | Pre-existing and out of scope for this design. Mitigated here by setting the key explicitly in all 15 modulated configs so no run depends on the fallback. Flagged to `bug-curator` in the handoff. |

### 2.5 Pre-registered predictions

Stated in advance, with the refutation criterion for each. Where the honest answer is "no
directional prediction", that is said rather than a direction being invented.

**H1 — the write site matters at all.**
- *Confirmed if*: the range of survival across the six target conditions (pooling input slices)
  exceeds 25 steps at matched environment experience.
- *Refuted if*: all six target conditions fall within 10 steps of one another.
- *Prior*: weak. The project has no controlled site comparison; this is the first.

**H2 — the sensory front-end is the right place to act.**
- *Confirmed if*: `t2enc` beats `t1none` by more than 25 steps **and** is not itself beaten by
  `t3rnn`, `t4act` or `t5crt` by more than 25 steps, at the best-performing input slice for each.
- *Refuted if*: `t2enc` fails to beat `t1none` by 10 steps while some other single-site target does.
- *Prior*: this project's Paper 1 has already committed to the front-end injection site as its one
  shared modulator architecture. **That commitment has never been tested against the alternatives**,
  which is precisely why this cell matters. A refutation here is a finding about the paper's plan,
  not a bad result.

**H3 — re-tuning the value estimator helps. Directional prediction: positive.**
- *Confirmed if*: `t5crt` beats `t1none` by more than 25 steps.
- *Refuted if*: `t5crt` fails to beat `t1none` by 10 steps, or is worse.
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

**H4 — more sites is better. No directional prediction.**
- Adding sites adds capacity, and each new site is initialised as a no-op (gain 1, offset 0), so at
  step 0 the all-four agent is exactly the single-site agent. That argument says adding sites cannot
  hurt *at initialisation*; it says nothing about the trained outcome, where four simultaneously
  adapting gain fields on a shared recurrent trunk could interact badly.
- *Recorded in advance*: `t16quad` landing below the best single-site target is **not** a surprise
  and is **not** evidence against modulation. Both directions are pre-accepted.

**H5 — the modulator must read the body. Directional prediction: I shows more internal-state
dependence than X.**
- *Primary evidence is behavioural, not survival* (§4.2). Confirmed if, at the same write target,
  the I arm's internal-state dependence of behaviour exceeds the X arm's, in the same direction, at
  **three or more of the five write targets**.
- *Refuted if*: X arms match or exceed I arms at three or more targets. That outcome would say the
  modulator's benefit, whatever it is, does not come from reading the body — which would be a
  serious problem for the project's mechanism story and must be reported as such.
- *Not evidence either way*: an ALL-vs-X difference (confound C2), or an I-vs-X difference in
  survival alone without a matching behavioural difference (a modulator can raise survival by
  route-tuning the encoder without producing any context-dependent policy).

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

### 2.7 Dependencies — why the configs are not written yet

This design deliberately stops short of producing YAML. Three things must land first.

| # | Dependency | State at time of writing | Blocks |
|---|---|---|---|
| **D-A** | **Part A of the site refactor** — the `agent.modulation.sites.{encoder,rnn,actor,critic}`, `agent.modulation.rnn_mechanism` and `agent.modulation.temperature.{enabled,clip}` keys, plus FiLM at the action and value heads. | **In implementation now.** The 12 existing neuromodulated configs are being migrated as this is written. | Factor 2 entirely. |
| **D-B** | **Part B of the site refactor** — the `agent.modulation.input_sensors` key (`"all"` or an explicit list of sensor names, resolved to indices against the run's own observation breakdown). | **Approved 2026-09-07; not yet built.** Note that [[MODULATION_SITE_REFACTOR]]'s status header still reads "Part B remains a proposal, NOT approved" — that header is stale and should be corrected by its owner. | Factor 1 entirely. |
| **D-C** | **A ruling on `return_mode: MC` semantics** — see confound C6. | **Unresolved.** Two project documents currently disagree. | The whole grid's budget and its comparability with the existing 5-seed baseline. |

Configs are produced in a **second pass**, once D-A and D-B are in the code and D-C has a ruling.
The §3.1 table below specifies exactly what that pass will write.

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

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | planned | `T1_none` | `rppo_nmnsite_t1none_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 2 | planned | `T2_enc_ALL` | `rppo_nmnsite_t2enc_ALL_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 3 | planned | `T2_enc_I` | `rppo_nmnsite_t2enc_I_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 4 | planned | `T2_enc_X` | `rppo_nmnsite_t2enc_X_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 5 | planned | `T3_rnn_ALL` | `rppo_nmnsite_t3rnn_ALL_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 6 | planned | `T3_rnn_I` | `rppo_nmnsite_t3rnn_I_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 7 | planned | `T3_rnn_X` | `rppo_nmnsite_t3rnn_X_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 8 | planned | `T4_act_ALL` | `rppo_nmnsite_t4act_ALL_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 9 | planned | `T4_act_I` | `rppo_nmnsite_t4act_I_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 10 | planned | `T4_act_X` | `rppo_nmnsite_t4act_X_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 11 | planned | `T5_crt_ALL` | `rppo_nmnsite_t5crt_ALL_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 12 | planned | `T5_crt_I` | `rppo_nmnsite_t5crt_I_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 13 | planned | `T5_crt_X` | `rppo_nmnsite_t5crt_X_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 14 | planned | `T16_quad_ALL` | `rppo_nmnsite_t16quad_ALL_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 15 | planned | `T16_quad_I` | `rppo_nmnsite_t16quad_I_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |
| 16 | planned | `T16_quad_X` | `rppo_nmnsite_t16quad_X_s42` | nmn_input_site_grid | pilot | 42 | — | — | — | — | — |

`wandb-job-type` is **`pilot`**, not `prod`, on every row. That is the honest label for a
single-seed screen and it keeps these runs from being pooled with production multi-seed studies in
any downstream query.

### 3.1 Configs to Produce (second pass — NOT yet written)

All 16 runs share one environment config, unmodified. Each run gets its own agent config, because
agent configs in this repo do not support `extends:` and are self-contained.

| Run | Config (env) | Config (agent) — to be written |
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
   then) `lr_modulator`, all at `0.0005`. See C5.
3. **`percept_add_bias_init: 0.0` stated explicitly** in all 15 modulated files, so no run depends
   on the fallback default noted in C12.
4. **`memory_bias_init` and `memory_clip` present** in all 15 modulated files even though they are
   unused under `rnn_mechanism: activation` — the refactor keeps them mandatory.
5. **Run 1 carries `modulation: {type: null}`** and no other modulation keys. Under
   `train.py:1138-1140` a null type collapses to no modulator at all, so none of the new mandatory
   keys is read. Its agent keys must be byte-identical to
   `configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml` so the lineage link to the existing
   5-seed record is exact (subject to C7).
6. **`env-config-reviewer` runs on all 16 files before launch**, with explicit attention to C5
   (learning rates), the input-sensor name spellings against
   `get_observation_breakdown`, and the sites/mechanism/temperature block.

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

### 4.2 Behavioural outcome — does the policy actually become context-dependent?

Survival alone cannot show context-dependent policy. A modulator can raise survival by statically
re-weighting sensory channels without ever conditioning behaviour on internal state. These measures
are what test the actual claim.

**A measurement rule this project has already been burned by.** An earlier analysis
([[INJURY_HIDING_SIGN_RECONCILIATION]]) found that the wounded agent's extra time in bushes is
mostly **freezing to heal**, not travelling to cover — and that the sign of the effect **flips with
the measurement window**. Across fourteen agents, "hides more when injured" was positive in 13 of 14
over the first 25 steps, in 2 of 14 over the first 100, and in 1 of 14 (by +0.003 percentage points,
i.e. zero) over the whole episode. The same analysis showed that once the measure is restricted to
steps on which the agent is **acting** rather than resting, cover use *falls* with injury in 14 of
14 agents in every condition. **Therefore no hiding metric is proposed here without stating both its
window and whether it conditions on acting.**

All three measures are computed offline from trajectory stores collected on `basic/04` — the
training world, not a purpose-built probe scene — binned by the **randomised starting injury** the
environment assigns before the agent acts, which is the only causally identified internal-state
variable available. Reported at **two windows, always both**: the first 25 steps (where the
randomised wound is still being felt, and the window the 14-agent reference record uses) and the
whole episode.

| ID | Measure | Definition | Why it is here |
|---|---|---|---|
| **B1** | **Rest rate by starting-injury quartile** | Fraction of steps on which the agent chose `Rest`, lightest versus heaviest quartile of randomised starting injury. | The dominant wound-driven behaviour in the existing record: +17.7 to +35.2 percentage points, in 14 of 14 agents. **Always reported with B2**, because B2 is uninterpretable without knowing whether the freeze response moved. |
| **B2** | **Bush occupancy conditioned on acting** | Fraction of steps in a bush, counting **only steps where the action was not `Rest`**, same quartile contrast. | The measure that separates travelling to cover from freezing on a square that happens to be a bush. Its baseline sign is established and negative: −0.77 to −1.79 points with no predator present, −1.24 to −9.49 with one, negative in 14 of 14 agents. |
| **B3** | **Internal-state dependence of the threat response** | The "hide more when a predator is near" effect, computed separately within each felt-wound bin, and the range of that effect across bins. Existing tool: `scripts/analysis/context_dependence.py --state felt_pain`, randomised block, early window. | The direct operationalisation of *context-dependent policy*: how much the response to an external cue varies with internal state. This is the primary evidence for H5. |

**Pre-registered behavioural thresholds.**

- **B2 sign flip is the headline behavioural prediction.** A modulated agent that produces genuine
  context-dependent cover use should move B2 **toward zero or positive** in the no-predator
  condition. A B2 value inside the 14-agent reference band of −1.8 to −0.8 percentage points is
  ruled "indistinguishable from the unmodulated behavioural baseline". A value at or above 0 is
  ruled a candidate context-dependence effect, subject to the same n = 1 caveat as everything else.
- **The authoritative null is Run 1**, this grid's own unmodulated control, not the 14-agent band.
  That band comes from a related but not identical world (the sensor-ladder configuration, which
  shares this environment's obstacle lineage and sensory settings but is a separately authored
  config) and is used only as an indicative prior. The numbers from the bush-refuge line are
  **not** a reference band here at all: that world's bushes physically block entry, which is a
  different mechanic.
- **A cheap power win, to be taken before launch.** The five existing unmodulated `MC` seeds
  already exist as finished runs. Collecting trajectories from all five gives a **5-seed dispersion
  estimate for B1/B2/B3 in exactly this configuration**, at no training cost. Under rule C7 those
  five are used for **dispersion only**, never for the level, if the training path changes before
  the grid launches. This should be done while they remain on the pre-change training path.

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
| 2 | The **I-versus-X contrast** at the two write targets that survive priority 1. | 5 each, so 20 runs | H5 is the project's mechanism hypothesis; it cannot rest on n = 1. |
| 3 | The **width-matched interoception control** (confound C3): `Extero Nociception` alone, 1 dim, against `Satiation` + `Interoceptive Nociception`, 2 dims, at the winning target. | 5 each | The only available way to separate "which information" from "how much information" using a name-keyed sensor selector. |
| 4 | Any arm that produced a **NaN or a value explosion** at a second seed (see §5, item 1). | 2 more | Distinguishes a bad run from an unstable site. |

**Not committed**: seeding all 15 modulated cells. That is 75 runs and roughly 900 GPU-hours to
answer questions most of which this screen will have already eliminated.

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

6. **Everything ties — all 15 modulated cells within 10 steps of the control.** This is a
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

---

## 6. Metrics and tooling requested

### 6.1 Metrics not currently logged

| Subfield | Content |
|---|---|
| **Metric** | **Per-step modulator signals in the trajectory store** — for each enabled site, the mean and standard deviation of the gain and of the offset across the 128 modulated neurons, recorded per step alongside the existing state columns (four floats per site per step). Unit: dimensionless gain / offset. |
| **Why now** | Hypothesis H5 and behavioural measure B3 ask whether the modulator's gain **tracks the internal state**. The training-time logs give only batch-wide means and standard deviations, which show *that* the gain varies but never *what it is a function of*. The trajectory store is where the state columns live, and it currently discards `mod_info` entirely — the model returns it and the collector drops it. Without this, the representational half of the project's 4x3 framework cannot be read for any run in this grid. |
| **Where it'd live** | `scripts/eval/traj_collect/traj_scan.py` (carry `mod_info` through the scan alongside `action`) and `scripts/eval/traj_collect/collect_trajectories.py` (add the columns to the parquet schema). |
| **Cost** | **Moderate.** The reduced form above is 4 floats per site per step — with four sites that is 16 extra float columns, comparable to the existing per-step state columns. A full per-neuron dump (128 values per site per step) would be **expensive** and is explicitly *not* what is requested; if a "which neurons" analysis is later needed it should be a separate, opt-in flag. |

### 6.2 Analysis tooling requested (not a new metric)

The `action` column already exists in the trajectory store, so the acting-conditioned outcome (B2)
needs **no new logging** — only an option in the existing analysis scripts to exclude `Rest` steps
from the bush-occupancy outcome, in `scripts/analysis/context_dependence.py` and
`scripts/analysis/hiding_drivers.py`. Listing it separately here to keep the distinction honest:
this is a several-line analysis change, not a training-time metric, and it does not gate the launch.

### 6.3 What is already covered and needs nothing

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

### B. Deviation from the standard template

This document reorders [training_analysis.md](../../../TEMPLATES/training_analysis.md) so that
everything pre-registered (§1, §2, §4, §5, §6) sits ahead of everything post-hoc (§7, §8, §9), and
the Analysis Plan appears **before** Results rather than after. §3 keeps its number and its column
semantics unchanged, because `training-runner` and `experiment-analyzer` both key on
`## 3. Launch Manifest`.

### C. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-09-07 | Initial pre-registered design. Configs deliberately not produced — blocked on dependencies D-A, D-B, D-C (§2.7). | `experiment-designer` |
