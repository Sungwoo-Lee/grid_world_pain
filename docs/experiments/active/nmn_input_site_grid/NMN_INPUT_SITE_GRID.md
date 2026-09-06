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
> **Amended**: 2026-09-07, in answer to `plan-reviewer`'s **NOT READY** verdict. The headline
> behavioural measure was replaced (§4.2), the resolution bands were raised (§2.3) and eight further
> findings were answered — §10 records what changed, and the two places where I did something other
> than what the review asked for, with the reasoning. Every amendment is still pre-registration: all of it was written **before**
> any run in this grid launched and before any of its data existed.
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
same configuration would behave the same way, and the gap between those two things is large. Both
numbers are reported side by side — the within-run interval and the five-seed control band measured
on the existing unmodulated runs (§4.2) — and a difference that clears the first but not the second
is not a result.

### 2.4 Confounds and limitations

| # | Confound | Severity | Assessment / mitigation |
|---|---|---|---|
| **C1** | **Single seed per cell.** | **Critical** | Not mitigated — it is the design. §2.3 states the resolution bands; §4.4 names the confirmation runs. Every verdict in §7-§9 must carry the n = 1 qualifier. |
| **C2** | **ALL and X differ by 8 dimensions, not 2** (both interoceptive channels *plus* six proprioception channels), because proprioception is deliberately in neither restricted slice. | **Critical for the ALL-vs-X reading** | Not mitigable within this design; it follows from the user's recorded decision. **The pre-registered rule: no ALL-vs-X difference is ever attributed to interoception.** The interoceptive contrast is I vs X, and only that one. |
| **C3** | **Input width is not matched across the input factor** (27 / 2 / 19). Any I-vs-X difference confounds *which* information the modulator reads with *how much*, and with the size of its input layer. | **High for H5** | Not mitigable with a name-keyed sensor selector. Stated in every H5 verdict. A width-matched follow-up is available and named in §4.4: `Extero Nociception` alone (1 dim) against `Satiation` + `Interoceptive Nociception` (2 dims). |
| **C4** | **Pinning `rnn_mechanism: activation` breaks continuity with the historical neuromodulated record**, every run of which used the legacy gate-bias operator on the memory site. | Medium | Deliberate. Uniformity across sites is what makes Factor 2 a single factor; continuity with a record that confounds site with mechanism is worth less. The trade is recorded so a future reader does not read cell `t3rnn` as a replication of past memory-site runs. |
| **C5** | **Learning rates.** Today one optimiser trains the shared trunk, both heads and the modulator at `lr_actor`, and the advertised `lr_critic` is read by nobody. If [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]] lands first, `lr_critic` and a new `lr_modulator` become live. | **Critical if unhandled** | **Pre-launch requirement**: all 16 configs must set `lr_actor = lr_critic = lr_modulator = 0.0005`, **cell 1 included**. `lr_critic` currently has no consumer, so this changes nothing about how these runs train today; it is a guard against the fix landing later and a rerun from cell 1's saved config silently training the critic at one fifth of the rate the rest of the grid used. It is the **one deliberate deviation** from byte-identity with `recurrent_ppo_cmp_mc.yaml` (which carries `lr_critic: 0.0001`) and is recorded as such in §3.1 requirement 5. Verified by `env-config-reviewer` before launch. (`plan-reviewer` finding 4.) |
| **C6** | **The meaning of `return_mode: MC` may change under the pending fix.** That fix's Part 1 proposes replacing the MC branch with "raw critic target, normalised advantage" — which is exactly the `MC_FIXED` mode that the later 25-run study measured as roughly an order of magnitude slower and 70 steps worse at matched experience. | **Critical, pre-launch blocker** | The two documents were written four days apart and have not been reconciled. **This grid must not launch until it is established which semantics `return_mode: MC` will carry.** If the fix lands as written, every arm here inherits a 10x experience penalty and the 10M-episode budget becomes far too small. Escalated to the user in the handoff; see §2.7. |
| **C7** | **Continuity of cell 1 with the five existing `MC` seeds** depends on no change to the training path between commit `788e5983` (those runs) and launch. | Medium | The site refactor promises the unmodulated path is bit-identical (its checks V1/C1). The pending fix does **not** — it changes loss values by design. **Pre-registered rule**: the five existing seeds are used for **dispersion** (how much seeds vary) and never for **level** (what the baseline is) unless the training path is confirmed unchanged. **Pre-registered sanity gate (the "C7 gate")**: cell 1 runs at seed 42 on the same environment, same agent config and same budget as the existing run `20260904-173804_rppo_cmp10m_mc_s42`, so it must **replicate** it — see §4.1 for the exact tolerance. If the unmodulated control does not reproduce the known result, the code path changed underneath the grid and **every** comparison in it is suspect, so the gate is checked first, before any hypothesis is scored. |
| **C8** | **Budget is in episodes, not environment steps, and better agents run longer episodes.** In the prior study total experience differed 3.3x across arms at a fixed episode budget. | **Critical** | Handled by design: the primary comparison is made at **matched environment steps** (the trainer logs `timesteps` directly), with the episode-axis number reported alongside. See §4.1. |
| **C9** | **Right-censoring.** Some arms may not have plateaued at 10M episodes. | High | Pre-registered exclusion rule in §5, item 4. |
| **C10** | **Checkpoints are not interchangeable across input arms.** Each slice changes the modulator GRU's input width, so the parameter trees differ. | Low — stated, not a defect | These are 16 fully independent trainings. No warm-starting, no shared initialisation beyond the common seed, no cross-arm checkpoint restore. Any attempt to restore across arms will fail loudly (the restore-completeness assertion), which is the desired behaviour. |
| **C11** | **The offline behavioural measures use a deterministic (argmax) policy** while the training-log survival series reflects the stochastic policy. They measure different objects. | Medium | Both reported side by side, never differenced. Same discipline as the prior study's C8. |
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

**H5-add (scored first — this is the load-bearing one).** *Directional prediction: an
interoception-reading modulator produces more injury-dependence of cover-seeking than no modulator
at all.*
- *Confirmed if*: at **three or more of the five write targets**, the I arm's injury-dependence of
  B0 exceeds **cell 1's** (the unmodulated control's) by more than the control's own 5-seed
  dispersion band, measured before launch on the five existing `MC` seeds (§4.2).
- *Refuted if*: at three or more targets, the I arms sit inside or below that control band. That
  outcome says a body-reading modulator adds no state-dependence beyond what the main network — which
  receives both body channels in every arm, control included — already produces on its own. It is
  the single most consequential negative this grid can return, and it must be reported as such
  rather than folded into an "inconclusive" summary.
- *Why this is the load-bearing test* (`plan-reviewer` finding 7): H5-spec below can be satisfied
  by a modulator that changes behaviour at all, because X's modulator cannot read satiation and
  reaches injury only indirectly. Only the comparison against the **unmodulated** control asks
  whether the modulator is contributing the state-dependence rather than merely correlating with it.

**H5-spec (the specificity check, scored second).** *Directional prediction: I exceeds X.*
- *Confirmed if*: at the same write target, the I arm's injury-dependence of B0 exceeds the X arm's,
  in the same direction, at **three or more of the five write targets**.
- *Refuted if*: X arms match or exceed I arms at three or more targets. That would say the
  modulator's benefit, whatever it is, does not come from reading the body — a serious problem for
  the project's mechanism story, to be reported as such.
- *Reported with its own caveat, always*: this contrast is close to true by construction, so a
  confirmation here **without** a matching H5-add confirmation is written up as "consistent with,
  but not evidence for, the mechanism".

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
  four moves). Both `Rest` (action 4) and `Eat` (action 5) are excluded — the denominator is "the
  agent decided to move", not "the agent was not resting". The `rested` column is used to verify the
  index convention per run rather than trusting the hard-coded 4.
- *Numerator*: those same rows with `agent_in_bush[t+1] == True`.
- *Injury binning*: by the **randomised starting injury** the environment drew before the agent
  acted — the only causally identified internal-state variable available — using the fixed edges
  already in `scripts/analysis/context_dependence.py` (`INJ_EDGES = [1e-9, 25, 50]`, giving bins
  *exactly 0* / *0-25* / *25-50* / *50+*). **Fixed edges, not quartiles**: quartiles are defined by
  the sample and would sit at different injury values in different arms, which would make the arms
  incomparable. Realised bin counts are reported.
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

`Δ_B0 = B0(top populated injury bin) − B0(bottom populated injury bin)`, in percentage points.

A bin needs at least 2,000 qualifying denominator steps to be used; otherwise the next bin inward is
taken and the substitution is reported.

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

#### The null band, measured before launch rather than assumed

**This is the pre-registration's load-bearing step and it happens before any grid run starts.**
`Δ_B0`, `B1`, `B2` and `B3` are computed on the **five existing unmodulated `MC` seeds**
(`rppo_cmp10m_mc_s42` through `s46`, already on disk with final checkpoints), which are the same
environment, the same agent configuration and the same budget as cell 1. That gives:

- the **level** of `Δ_B0` for an unmodulated agent in this exact world — the null that H5-add is
  scored against; and
- its **5-seed dispersion**, the only estimate this project will have of how much a one-seed
  behavioural number can move for reasons that have nothing to do with the modulator.

**The control band** is the min-to-max range of `Δ_B0` across those five seeds, per window and
predator condition. A grid arm counts as exceeding the control only if its `Δ_B0` lies **above the
top of that band**, in the predicted positive direction. Under rule C7, if the training path is
confirmed unchanged the band is used for both level and dispersion; if it is not, the band is used
for **dispersion only** and cell 1 supplies the level, with the band's width carried over.

Two numbers accompany every arm and are never conflated: the **within-run** binomial 95% confidence
interval on `Δ_B0` (how precisely that one agent has been characterised — it will be narrow) and the
**five-seed control band** (how much a second seed of the *same* configuration could move — it will
be much wider). A difference that clears the first but not the second is not a result.

#### Checkpoint rule

Every behavioural measure is computed at **two checkpoints per run, both reported**
(`plan-reviewer` finding 8):

1. the checkpoint whose logged environment-step count is **nearest the common matched budget** used
   for the primary survival reading (§4.1) — this is the comparison that is not confounded by the
   episode-budget artefact C8, under which arms differ in total experience by up to 3x; and
2. the **final** checkpoint of each run, for continuity with how every previous behavioural analysis
   in this project was collected.

Where the two disagree, the matched-budget reading carries the verdict and the disagreement is
reported. The five baseline seeds are collected at their final checkpoints before launch (for the
null band) and re-collected at their nearest-to-matched checkpoints once the common budget is known;
they save a checkpoint every 200,000 episodes, so a nearby one always exists.

#### Collection parameters

Pre-registered so the collections are comparable: **300,000 episodes per checkpoint**, on
`basic/04` — the training world, not a purpose-built probe scene — with a **shared `seed_base`
across every run in the grid and every baseline seed**, which makes all arm-to-arm comparisons
**paired** on identical episode draws. This matches the sensor-ladder study's collection size, whose
stores are on disk as the precedent, at roughly 2 GB per collection. (The first version of this
design said "roughly 200 evaluation episodes per run"; that was simply wrong and is corrected here
and in §2.3.) The policy is deterministic (argmax) at collection time, which is
confound C11.

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
small additions to existing analysis scripts, and the first must exist before any behavioural
verdict is read.

**(a) The bush-entry rate (B0) — needed before the null band can be measured.** Every column it
requires already exists in the trajectory store (`agent_in_bush`, `action`, `rested`, `agent_row`,
`agent_col`, `animal_row`, `animal_col`, `animal_active`, plus the episode table's starting draws),
so this is arithmetic over data already on disk, not new logging. What is needed is the measure
itself — the transition-based denominator of §4.2 — added to
`scripts/analysis/context_dependence.py` alongside the existing occupancy measures, with the
starting-injury binning and the `NEAR_D = 2` predator condition it already implements. Roughly a
few dozen lines. **Priority: this runs on the five existing `MC` seeds before launch**, because the
null band is what makes the pre-registered behavioural threshold meaningful; a threshold with an
assumed null is exactly what finding 3 objected to. The demoted B2 needs the same scripts' existing
Rest-exclusion option, which also does not yet exist.

**(b) The critic-value pre-check for H3 (§2.5).** Plot the distribution of critic values conditioned
on injury, on the existing `MC` checkpoints. The value output is computed inside
`scripts/eval/eval_rollout.py` (`get_action_and_value_nnx`) and discarded, and the trajectory store
deliberately does not record it, so this needs a short script that replays stored observation
sequences through a loaded checkpoint's critic — recurrent, so per-episode sequential replay. Its
result **does not gate the launch and does not move any H3 threshold** (§2.5); it changes only
whether an H3 null reads as expected or as surprising.

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

**(b) The injury breakdown uses fixed bin edges, not quartiles.** The instruction was to break B0
down by starting-injury **quartile**. Quartiles are defined by the sample, so the boundary between
"lightly wounded" and "badly wounded" would sit at a different injury value in each of the sixteen
arms, and the arms would no longer be comparable on the axis the whole measure is about. §4.2
therefore uses the fixed edges already implemented in `scripts/analysis/context_dependence.py`
(exactly 0 / 0-25 / 25-50 / 50+), which are identical across arms, already tooled, and already the
binning the fourteen-agent reference record used. Realised bin counts are reported so a reader can
see the resulting balance. The contrast is still top-bin minus bottom-bin, exactly as a quartile
contrast would be.

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
| Movement actions are 0-3; `Rest` is 4, `Eat` is 5 | `src/environment/core.py:580-581,660-661` | `rested = (action == 4)` when the rest action is enabled; `eat_action_idx = 5` under the same condition. The store's `rested` column re-verifies this per run. |
| Predator-near threshold: Chebyshev distance 2 | `scripts/analysis/context_dependence.py:53` (`NEAR_D = 2`) | Reused unchanged so B0 and the existing occupancy measures share one definition of "near". |
| Starting-injury bin edges 0 / 0-25 / 25-50 / 50+ | `scripts/analysis/context_dependence.py:50` (`INJ_EDGES`) | Same edges the fourteen-agent reference record used; fixed rather than sample-defined so the sixteen arms are comparable. See §10.1(b). |
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
