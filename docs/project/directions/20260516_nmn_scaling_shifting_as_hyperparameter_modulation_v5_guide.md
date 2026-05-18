---
title: "NMN direction v5 — a plain-English reader's guide"
status: draft
audience: user, anyone-new-to-project
last_updated: 2026-05-18
companion_to: docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md
related:
  - docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md
  - docs/pi/calls/2026-05-16_impact_vs_reasonable_call.md
  - docs/project/concepts/film_neuromod_integration.md
  - docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md
---

# NMN direction (v5) — a plain-English reader's guide

This guide is for a reader who has not read v5. It explains what v5 says, why it says it, and what decisions sit downstream — without the symbols, run IDs, and cross-references that v5 unavoidably accumulates. Read this in ~10 minutes; open v5 only for the sections you want to deepen.

## 1. What the project is asking

Brains run on a small number of *neuromodulator* chemicals — noradrenaline (NA), acetylcholine (ACh), dopamine (DA), serotonin (5-HT). Each is broadcast from a small "source nucleus" to large parts of the brain at once, and each *changes how the rest of the network behaves*: how fast it learns, how sharply it commits to a choice, how strongly it weights one input over another. This is *neuromodulation*: **one signal, many downstream effects, distributed broadly**.

This project asks: **can a reinforcement-learning (RL) agent be built so that a small side-network — we call it the NMN, *Neuromodulatory Network* — plays the same role? One side-signal that fans out through the agent and changes its behaviour in multiple ways simultaneously?**

## 2. The substrate, in one paragraph

The main RL agent is a recurrent neural network trained with PPO (a standard RL algorithm). The NMN is a *small* side network that watches the main network's internal state and emits, for each feature, two numbers: a multiplicative *scale* $\gamma$ and an additive *shift* $\beta$. These are injected into the main network at three places — *encoder* (perception), *GRU update gate* (memory), *policy logits* (action selection) — through the **FiLM** operator (feature-wise linear modulation):

$$h \mapsto \gamma(c) \odot h + \beta(c)$$

That is the whole architectural idea. One small recurrent state inside the NMN (`mod_h`) produces, through three different *readout matrices*, three target-specific patterns of γ and β. Same upstream signal; different downstream effects at each site.

## 3. The headline claim, in one sentence

A single side-signal, fanned out through *target-specific readout matrices*, can produce a *basket of behavioural-knob-like effects* at each target — instead of one signal per effect.

The paper claims this is a *possibility-of-bridge* result, not a proof: one demonstration end-to-end. It does *not* claim FiLM is THE substrate of biological neural gain, and it does *not* claim every neuromodulator paper fits this picture.

## 4. The biological exemplar we're targeting

v5 commits to one biological exemplar: **noradrenaline from the locus coeruleus (LC-NA)** — the brain's small brainstem nucleus that broadcasts NA broadly across cortex. We pick LC-NA because the project already has an empirical finding (called the "R2 anchor") that fits its classical signature:

- The agent with the NMN recovers about 25× faster than the baseline when it returns to a previously-seen task stage.
- The NMN's internal signal swings *sign-consistently* at three of four task-stage transitions.

That brief-burst-at-regime-change pattern is the textbook signature of LC-NA in the *adaptive-gain theory* (Aston-Jones & Cohen 2005): phasic bursts at moments of regime change, quiet during steady state.

The user's binding scoping rule for the paper is: **target one system explicitly, acknowledge that it produces many effects, defer full coverage of the other neuromodulators to future work.** Trying to map four neuromodulators one-to-one onto four distinct RL hyperparameters is, in v5's reading, mathematically impossible — real biological function is overlapped and distributed.

## 5. Why the three injection sites are *targets*, not "channels"

v4 of the direction memo said *"site A is the ACh channel; site C is the NA channel"*. The 2026-05-16 four-professor symposium found that this was the *same one-to-one mistake at a finer scale*. v5 drops it.

Instead: each site is a *target* on which the same upstream LC-NA-analog signal acts through its own *target-specific readout matrix*. Each target produces a *basket of plausible effects*. For example, at site C (the policy target), the basket simultaneously contains:

- *temperature-like* effects (how sharply the agent commits to an action),
- *action-prior-shift-like* effects (which action it tends to pick),
- *attentional-bias-like* effects,
- *policy-precision-like* effects,
- *exploration-bonus-like* effects.

The basket comes from one signal acting through one readout. The substrate is *not* claiming each effect is a separate channel; it is claiming they co-arise.

**The biological analogue is *receptor density*.** One LC projection reaches many cortical areas; what it *does* at each area depends on which adrenergic receptors that area expresses. The FiLM readout matrix is the in-silico analogue of receptor density.

## 6. Why this architecture, mathematically

FiLM lives inside a well-defined family:

$$\text{FiLM} \;\subset\; \text{Hypernetwork} \;\subseteq\; \text{Bayesian Hypernetwork (point-mass limit)}$$

A *hypernetwork* is a network that emits weights for another network as a function of input context. The project's three FiLM readouts are exactly this — the NMN is the "context generator", the three γ/β pairs are the "emitted weights" at each target.

The Galanti & Wolf 2020 modularity theorem says, in plain English, that the *right* architectural shape for "one signal, many target-specific effects" is exactly this — a hypernetwork with multiple target-specific readouts. **The dissociation between effects lives in the readouts, not in the upstream signal.** So the project's architecture is not a workaround; it is the textbook architectural form for the claim we are making.

A future upgrade (called "Row EE-6") adds two more pieces — a small ensemble of FiLM heads (Turkoglu 2022) plus a heteroscedastic uncertainty head (Kendall & Gal 2017) — which moves the substrate from "deterministic hypernetwork" toward "approximate *Bayesian* hypernetwork with heteroscedastic likelihood". That upgrade is *not* in v5's headline; it is a candidate paper framing (see §10 below, Path B).

## 7. What we explicitly do *not* claim

- We do not claim FiLM is *the* substrate for biological neural gain. We claim *a* possibility.
- We do not claim to recover all four Doya RL-hyperparameter channels. In particular, the *discount factor* γ (sometimes mapped to 5-HT) **cannot** live inside a forward-pass FiLM operator — it lives at the TD-target loss, mathematically. v5 follows Lee 2024 in keeping it out.
- We do not claim coverage of ACh, DA, or 5-HT systems in this paper. They are named as future work.
- We do not claim the NMN proves *pain content*. A separate four-condition gate (§9 below) decides whether the behavioural signature earns the word "pain-like" rather than "nociception-like".

## 8. How v5 proposes to test the claim

Five families of test, all on the same agent on the same trials:

1. **The headline falsifier (at site C).** Run the agent in four arms: free, γ frozen, β frozen, both frozen. Measure recovery speed, entropy spike at task transitions, and how strongly the *surviving* arm tracks the value-prediction error $|\delta|$. Three readable outcomes (called U / V / T) discriminate three different mechanistic stories of what the modulator actually does.
2. **Per-target deficits (3 × 3 dissociation).** Freeze one site at a time; measure which subtest each freeze most damages. The 3 × 3 deficit matrix should be *diagonal-dominant* — each target preferentially carries its own basket of effects. This is the receptor-density-as-readout signature.
3. **Two-timescale tests (T/P split, prerequisite).** Split the NMN's single recurrent state into two — a fast (*phasic*) one and a slow (*tonic*) one. Each timescale should drive a *basket* of effects simultaneously, not one effect each. This is the strongest multifunctionality test the testbed supports.
4. **Architecture-discrimination probes.** Measure signatures that distinguish FiLM from competing architectures. The key one: under Rodriguez-Garcia 2026's gradient-level gain, the raw-weight Hessian gets *amplified* by $g^2$; under forward-pass FiLM, it should be untouched. This is one of the cleanest discriminators we have.
5. **The pain-construct gate (four behavioural dissociations).** Before the signature earns the label "pain-like", four behaviourally-defined dissociations must hold: channel-selective hypervigilance, time-locking to injury inference, joint perceptual–mnemonic–policy reweighting, and dissociation from a slow-input EMA baseline. If any fails, the paper retreats from a pain-content claim to a substrate-only claim.

## 9. The four interpretive lenses on the same data

v5 lays out four *complementary* readings of the same substrate. None is load-bearing on its own; each illuminates a different aspect, and where they disagree the data can discriminate.

| Lens | Reading of the modulator |
|---|---|
| **Doya-channel** | ACh-like at perception, NA-like at policy, plasticity-gating at memory |
| **Precision-coding** | Sensory / state-transition / policy *precisions* (one signal, many precisions) |
| **Hypernet / BHN** | One conditioner, three target-specific readouts, identifiability via Galanti–Wolf |
| **Target-flexibility (biology)** | One nucleus, many downstream effects via receptor-density readouts |

The substrate-level claim sits *under* all four lenses, not inside any one.

## 10. The user's next decision — four candidate paths

v5's body is *path-neutral*. It supports four different framings for the paper itself. The user picks one; v5 stops being rewritten after that. The PI ranks them under two readings:

| Path | One-line headline | Code work | Time to submit | Venue family |
|---|---|---|---|---|
| **A** (PI top pick) | "An LC-NA-analog modulator demonstrates *one signal → many behavioural effects* in an RL agent." | none | 2–4 months | comp-bio / comp-neuro |
| **B** | "First approximate *Bayesian Hypernetwork* with heteroscedastic likelihood for an RL policy." | Row EE-6 (4–6 weeks) | 3–5 months | NeurIPS / ICLR / TMLR |
| **C** | Path A + soft pain-construct upgrade conditional on a probe-trial extension landing. | A + probe-trial harness + EMA baseline | 4–7 months | comp-bio default; *PAIN* if probe lands |
| **E** | Path A's narrowest single-claim version. | none | 1.5–3 months | comp-bio + workshop fallback |

- Under the *"what the project's identity is"* reading: **A > E > B > C**.
- Under the *"what the field rewards"* reading: **B > A > E > C**.

## 11. A glossary, on one screen

- **NMN** — Neuromodulatory Network. The small side network whose hidden state is `mod_h`. The in-silico analogue of a neuromodulator source nucleus.
- **FiLM** — feature-wise linear modulation. $h \mapsto \gamma \odot h + \beta$.
- **Sites A / B / C** — the three injection points: encoder pre-fusion (*perception*), GRU update gate (*memory*), policy logits (*action selection*).
- **LC-NA** — locus coeruleus → noradrenaline. The biological exemplar v5 commits to.
- **Doya channels** — the classical RL-hyperparameter mapping (DA / ACh / NA / 5-HT ↔ TD-error / learning-rate / temperature / discount). v5 does *not* commit to this one-to-one; it uses it only as one of four interpretive lenses.
- **T/P split** — the planned architectural change giving the NMN two recurrent states with separable phasic (P) and tonic (T) timescales. Prerequisite for the multi-effect-by-timescale test.
- **Row EE-6** — the FiLM-Ensemble + heteroscedastic-precision compound. The architectural upgrade that promotes the substrate toward an *approximate* Bayesian Hypernetwork. In motivation for Path A; headline for Path B; deferred for Path E.
- **R2 anchor** — the project's existing empirical finding (faster recovery on returns to known stages, with phasic-burst signature at transitions). The reason LC-NA is the natural target.

## 12. Where to read more

- **The full v5 memo** — [`20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md`](20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md). The body for everything this guide compresses.
- **The PI call (path options)** — [`docs/pi/calls/2026-05-16_impact_vs_reasonable_call.md`](../../pi/calls/2026-05-16_impact_vs_reasonable_call.md).
- **The symposium memos** — [`docs/project/symposium/20260516_impact_vs_reasonable/`](../symposium/20260516_impact_vs_reasonable/). Four professors' contributions plus the postdoc synthesis.
- **The architectural concept memo** — [`docs/project/concepts/film_neuromod_integration.md`](../concepts/film_neuromod_integration.md). Where the FiLM-as-bridge integration lives mathematically.
- **The lineage investigation** — [`docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md`](../concepts/film_hypernet_bnn_lineage_and_critic_modulation.md). Where FiLM sits in the family tree.
