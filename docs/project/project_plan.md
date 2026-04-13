---
title: GridWorld Pain — Project Plan
last_updated: 2026-04-12
---

# GridWorld Pain — Project Plan

> This is the main project context document. It records the goal, the
> target behavior, the central research issue, and the staged plan for
> resolving it. Other documents in `docs/develop/` provide the technical
> depth behind each phase.

## 1. Project Goal

**GridWorld Pain** is a reinforcement learning research platform built to
**simulate pain in a grid world at a level of complexity sufficient to study
its cognitive and behavioral consequences**. The aim is not merely to attach a
"negative reward" signal to damage events, but to construct an environment and
agent rich enough that *pain-like phenomena emerge* as a result of
interoceptive state, sensory processing, and decision-making interacting over
time.

Concretely, the project asks:

> Given an embodied RL agent with homeostatic drives (hunger, satiation,
> injury) and multimodal exteroception (nociception, olfaction, vision,
> collision, proprioception, location), can we build an environment and an
> architecture in which the agent develops *pain-like behavior* — not by
> hand-coded rules, but as an emergent consequence of optimizing survival?

The grid world is therefore designed with enough mechanical depth — predators,
dangers, regenerating resources, injury smoothing, metabolic cost, homeostatic
reward — to create genuine conflicts between competing needs, rather than a
toy damage signal.

## 2. Target Behavioral Signatures

The project is considered successful to the extent the agent exhibits
**pain-like behavioral signatures**, defined here as two interacting phenomena:

### 2.1 Hypervigilance

After nociceptive experience, the agent should become **more sensitive to
threat-related sensory cues**: amplifying attention toward dangers, retaining
threat information across time, and adopting more cautious action policies.
This mirrors the clinical signature of acute/chronic pain states, which
include attentional bias toward threat, ruminative memory of harm, and
behavioral avoidance.

Hypervigilance is explicitly *not* a single-channel phenomenon. It is expected
to appear simultaneously across perception, memory, and decision-making, and
to persist beyond the immediate injury — resembling maladaptive chronic pain
when it outlives the original insult.

### 2.2 Conflicting-Needs Modulation

A pain signal that simply dominates the reward function produces a trivially
avoidant agent. The interesting case is when pain must be **weighed against
competing homeostatic drives** — hunger, satiation, exploration — so that the
agent must dynamically decide when to endure risk for food and when to
withdraw.

The environment is therefore tuned so that food and danger can co-occur, that
starvation and injury are both terminal, and that the agent cannot always
satisfy one drive without exposing itself to the other. Pain-like behavior, in
this framing, is the *modulation* of ongoing goal-directed behavior by
interoceptive state, not its suppression.

Together, hypervigilance and conflicting-needs modulation form the behavioral
target of the project.

## 3. Approach: Whole-Network Neuromodulation

To produce these behaviors without hand-engineering them, the project adopts a
**neuromodulation algorithm that modulates the entire network**, rather than
patching a single layer.

The motivating observation is biological: neuromodulators (acetylcholine,
noradrenaline, dopamine, serotonin, opioidergic systems) do not act on a
single cognitive domain. They simultaneously adjust sensory gain, memory
persistence, exploration drive, and reward sensitivity, producing the
*coordinated* shifts in state we recognize as affective tone. Restricting
modulation to a single site (e.g. perceptual precision alone) cannot
reproduce the full pain syndrome, which is jointly perceptual, mnemonic, and
behavioral.

Accordingly, the architecture centers on a **single recurrent neuromodulatory
core** — driven by the full observation vector including interoceptive
channels — whose outputs are injected into multiple sites of the main agent
network:

- **Perception (A):** gain and threshold modulation on the sensory encoder,
  implementing precision-weighting of multimodal features.
- **Memory (B):** bias on the recurrent update gate, controlling how strongly
  past state (e.g. prior threats) is retained versus overwritten.
- **Decision-making (C):** modulation of action entropy (PPO) or imagined
  reward scaling (DreamerV3), producing risk-sensitive exploration and
  planning.

Because all injection sites are driven by a *shared* slowly-evolving
neuromodulatory state, the resulting modulation is coherent across domains:
perception, memory, and action shift together in response to the agent's
interoceptive history. This whole-network scope is what allows hypervigilance
and conflicting-needs modulation to emerge as joint phenomena rather than
isolated tricks of any single layer.

## 4. Main Issue: Neuromodulation Does Not Outperform the Baseline

The central blocker for the project is empirical: **across the diagnosis
series [NMN_PERFORMANCE_DIAGNOSIS_v1–v8](../develop/), the neuromodulated
agent does not reliably outperform an unmodulated baseline on survival, with
or without perceptual noise.** v8 is the clearest statement of the null
result — MC-return FiLM at its best configuration matches the LayerNorm
baseline within noise, and GAE-FiLM is measurably worse. The whole-network
modulation scope described in §3 is architecturally in place, but the signal
it is supposed to carry is not being learned.

Four interacting causes are implicated:

1. **No precision training signal.** The RL loss rewards *behavior*, not
   *calibrated gating*. Nothing in the PPO/Dreamer objective pushes the
   modulator to down-weight unreliable channels, so γ collapses to near-
   identity and the modulator is effectively bypassed.
   ([PRECISION_MODULATION_ARCHITECTURE.md](../develop/PRECISION_MODULATION_ARCHITECTURE.md))
2. **Static FiLM cannot express per-timestep reliability.** FiLM applies
   learned (γ, β) *uniformly* across time — it cannot distinguish "this
   channel is noisy right now" from "this feature should always be scaled
   this way," which is exactly the computation precision-weighting requires.
   ([NMN_PERFORMANCE_DIAGNOSIS_v8.md](../develop/NMN_PERFORMANCE_DIAGNOSIS_v8.md),
   [FiLM_ENSEMBLE_SENSORY_PRECISION.md](../develop/FiLM_ENSEMBLE_SENSORY_PRECISION.md))
3. **Noise landscape does not reward precision.** The current perceptual
   noise profile is too uniform across modalities to make selective gating
   pay off: there is no strong reliability contrast between channels, so an
   unmodulated agent can do about as well as a modulated one. The task has
   to be reshaped so that precision-weighting is actually required for
   survival — and, correspondingly, so that an unmodulated agent's
   performance is measurably *degraded* by noise it cannot filter.
   ([NMN_PERFORMANCE_DIAGNOSIS_v8.md §6.2](../develop/NMN_PERFORMANCE_DIAGNOSIS_v8.md),
   [PRECISION_MODULATION.md](../develop/PRECISION_MODULATION.md))
4. **Temperature saturation and critic instability.** The PPO temperature
   head saturates at the clip ceiling across MC-FiLM runs, rendering that
   injection site non-functional; under GAE, the modulator actively
   destabilizes the critic under noisy inputs rather than helping it.
   ([NMN_PERFORMANCE_DIAGNOSIS_v8.md §5.1](../develop/NMN_PERFORMANCE_DIAGNOSIS_v8.md))

### Direction for Solving the Issue

The plan is deliberately **staged and conservative**: each phase has to
clear a concrete gate before the next begins, so that we never stack an
architectural change on top of an environment that does not reward it, and
never blame an architecture for a task that gives it nothing to learn from.
The two leading success gates are fixed and drive the whole plan:

- **G1 — Noise creates headroom.** An unmodulated LayerNorm baseline's
  survival must drop *measurably and reproducibly* under the new noise
  profile compared to the no-noise condition. If noise does not hurt the
  baseline, there is no room for precision modulation to help, and no
  phase beyond G1 is meaningful.
- **G2 — Emergent hypervigilance signature.** The modulated agent must show
  a **time-locked, cross-domain** response to injury — post-injury
  perceptual-gain shift, memory-gate bias toward retention, and an action
  policy shift (temperature drop in PPO, reward-scale drop in DreamerV3) —
  rather than an isolated change on any single injection site. The
  cross-correlation of these signals across a shared modulatory state is
  the empirical fingerprint of the hypothesis in §2.

Both gates are anchored in
[NEUROMODULATION_ALGORITHM.md](../develop/NEUROMODULATION_ALGORITHM.md) and
the [NMN_PERFORMANCE_DIAGNOSIS_v8](../develop/NMN_PERFORMANCE_DIAGNOSIS_v8.md)
null result; they are not negotiable inside this plan.

#### Phase 1 — Reshape the noise landscape until the baseline bleeds

**Objective:** Make the task actually *demand* precision weighting.

The environment already supports everything Phase 1 needs — state-dependent
noise, per-modality σ / mode / clip, and injury-scaled noise. The work is
tuning, not engineering:

- **Heterogeneous per-modality noise** with sharp reliability contrast
  across channels (e.g. high-variance olfaction and visual, low-variance
  proprioception and collision). Uniform noise flattens the landscape that
  a precision head would otherwise exploit.
- **State-dependent, injury-scaled noise** on threat-relevant channels
  (olfaction, nociception, visual) so that *reliability itself* becomes
  time-varying and correlated with interoceptive state — this is the
  signal a shared recurrent neuromodulatory core can latch onto, per
  [NEUROMODULATION_ALGORITHM.md §1.4](../develop/NEUROMODULATION_ALGORITHM.md)
  hypotheses H1–H5.
- **Magnitudes tuned against G1.** Sweep σ_base and injury_scale on the
  LayerNorm baseline (no modulation) across ≥3 seeds; pick a profile where
  baseline survival drops by a meaningful margin vs the noise-free
  condition and the drop is seed-stable. This profile becomes the *canonical
  noise preset* used from Phase 2 onward.

**Exit gate:** G1 passed on the LayerNorm baseline.

#### Phase 2 — Reproduce the null, then probe the FiLM variants

**Objective:** Confirm, under the canonical noise preset, that the v7/v8
null result still holds — and isolate which FiLM variant, if any, is
closest to breaking it before we spend effort on a precision head.

- Re-run the unmodulated baseline and the FiLM variant(s) catalogued in
  [FILM_MODULATION_PLAN.md](../develop/FILM_MODULATION_PLAN.md) and
  [FiLM_PAPERS_REVIEW.md](../develop/FiLM_PAPERS_REVIEW.md) — at minimum
  Multiplicative, PreActivation, FiLM (LayerNorm-targeted), and
  FiLMNoNorm — under identical canonical noise, seeds, and horizon.
- Record per-variant γ / β distributions, gate health, temperature
  trajectories, and hypervigilance probes (see G2) even when survival is
  flat; the goal here is *diagnostic*, not victory.
- Honest expected outcome: the null result replicates. What we want is a
  clean characterization of *why* each variant fails (collapse to
  identity, temperature saturation, critic destabilization) on the
  canonical task, which becomes the prior for Phase 3.

**Exit gate:** A characterized null (or unexpected partial win) on
canonical noise, with at least one variant identified as the least
pathological starting point for Phase 3.

#### Phase 3 — Add the precision training signal

**Objective:** Give the network direct gradient pressure to *estimate
per-channel reliability*, not only to act well. This is where we move
beyond the purely RL-driven gating of Phases 1–2.

- **Precision head.** On top of the chosen Phase-2 starting-point variant,
  add a precision head (per-modality log-σ̂²) driven by the modulator's
  shared GRU hidden state. Train it with a **heteroscedastic auxiliary
  loss** (Kendall & Gal / FiLM-Ensemble formulation from
  [FiLM_ENSEMBLE_SENSORY_PRECISION.md](../develop/FiLM_ENSEMBLE_SENSORY_PRECISION.md))
  on observation reconstruction or next-step prediction, with the loss
  weight `lambda_precision` as a first-class tunable.
- **Gate injections on learned precision.** Existing Injection A (encoder
  FiLM) combines its static (γ, β) with the learned per-channel π̂ from
  the precision head, so that down-weighting a channel is driven by its
  estimated reliability, not a fixed parameter. Injections B/C remain
  driven by the same shared recurrent state, preserving the
  coordinated-modulation hypothesis (H4).
- **DreamerV3 variant.** In parallel, explore the world-model-decoder
  route sketched in
  [FiLM_ENSEMBLE_SENSORY_PRECISION.md §6](../develop/FiLM_ENSEMBLE_SENSORY_PRECISION.md):
  heteroscedastic decoder → per-modality log-precision → gate the
  DreamerV3 encoder/actor. This is *not* a replacement for the PPO
  precision head; it is a cross-check that the effect is not
  algorithm-specific.

**Exit gate:** Both G1 and G2 hold on the canonical noise preset, across
seeds. Specifically: the modulated agent beats the baseline by a
seed-stable margin *and* the post-injury hypervigilance fingerprint is
detectable, time-locked, and cross-correlated across injection sites.

#### Phase 4 — Hypervigilance as the primary scientific readout

Once G1 + G2 hold, the project pivots from "does it work" to "what is it
showing us." Phase 4 analyses — hypothesis H1–H5 in
[NEUROMODULATION_ALGORITHM.md §1.4](../develop/NEUROMODULATION_ALGORITHM.md) —
are only meaningful after a credible precision-weighted modulator exists,
which is why they sit after Phase 3 rather than in parallel with it:

- Time-locked analysis of γ, memory-gate bias, and action/reward
  modulation around injury events.
- Cross-correlation of the three injection sites to test the
  coordinated-modulation hypothesis (H4).
- Modulator-GRU-timescale sweep (`mod_hidden_size`) to test whether
  hypervigilance *outlasts* injury recovery, the chronic-pain analog
  (H5).

Phase 4 is where "pain-like behavior" stops being a design goal and
becomes an observable of the trained system.

---

### How this document is used

This file is the **main context document** for the project — it should be
read before any work on env tuning, modulator architecture, or training
analysis. It deliberately does not commit to exact hyperparameters,
WandB run names, or file-level diffs; each phase above spawns its own
issue-plan or training-analysis doc in
[docs/develop/](../develop/) (per the templates in
[docs/TEMPLATES/](../TEMPLATES/)) where the tactical details live. When a
phase's exit gate is cleared, cross-link the resulting analysis back to
this plan so that the project's state is always derivable from this
single entry point.

## 5. Key References

Technical depth behind each phase lives in `docs/develop/`:

- Environment mechanics and sensor layout:
  [ENVIRONMENT_SUMMARY.md](../develop/ENVIRONMENT_SUMMARY.md)
- Four-injection-site neuromodulation, hypotheses H1–H5, hypervigilance
  validation plan:
  [NEUROMODULATION_ALGORITHM.md](../develop/NEUROMODULATION_ALGORITHM.md)
- FiLM variants and injection strategies, staged catalog of modulators:
  [FILM_MODULATION_PLAN.md](../develop/FILM_MODULATION_PLAN.md),
  [FiLM_PAPERS_REVIEW.md](../develop/FiLM_PAPERS_REVIEW.md)
- Ensemble + heteroscedastic precision loss, decoder-side proposal:
  [FiLM_ENSEMBLE_SENSORY_PRECISION.md](../develop/FiLM_ENSEMBLE_SENSORY_PRECISION.md)
- Precision-gated modulation architecture (PPO + DreamerV3 routes):
  [PRECISION_MODULATION_ARCHITECTURE.md](../develop/PRECISION_MODULATION_ARCHITECTURE.md),
  [PRECISION_MODULATION.md](../develop/PRECISION_MODULATION.md)
- Null-result diagnosis series (v1–v8), most recent:
  [NMN_PERFORMANCE_DIAGNOSIS_v8.md](../develop/NMN_PERFORMANCE_DIAGNOSIS_v8.md)
