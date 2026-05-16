---
title: "Symposium contribution — RL/Bayesian-DL/architecture position on impactful-vs-reasonable framing"
session: 2026-05-16_impact_vs_reasonable
contributor: professor-rl-bayesian-dl
date: 2026-05-16
inputs:
  - docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md
  - docs/project/concepts/film_neuromod_integration.md
  - docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md
prior_contributions:
  - docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation.md (v1 §8)
  - docs/project/references/neuromodulatory_algorithms/feedback/20260516_rl_bayesian_dl_predictions_for_v3.md (v3 round)
  - docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md (investigation memo)
---

# Symposium contribution — RL/Bayesian-DL/architecture position on impactful-vs-reasonable framing

## Plain-English entry point

The user has called this symposium because the project sits at a portfolio-level crossroads. After four revisions of the headline direction memo (v1 → v4) plus the FiLM-and-neuromodulator integration memo and the lineage investigation, the paper-shape decision has converged on a small handful of choices that differ in two dimensions: **how strongly the paper claims a bridge between neural-gain mechanisms (the cellular operation of multiplicatively scaling a neuron's input-output curve) and behavioural-RL-hyperparameter modulation (Doya 2002's mapping of biological neuromodulators to scalar knobs of an RL algorithm — learning rate, inverse temperature, discount factor), and how much architectural / experimental commitment the project takes on to support that claim**. The user has also added two constraints since v4: (1) the paper should suggest **possibility**, not full coverage — a "this bridge can be built; here is one instance of it" thesis rather than a "neural gain IS Doya channels" thesis; (2) the paper should explicitly **embrace one-to-many** mapping — a single biological neuromodulator influences multiple downstream computations, not one — and not silently inherit Doya 2002's one-to-one channel attribution.

My architectural-side position is: **v4 is technically clean but rhetorically still over-committed under the new "possibility + one-to-many" constraint**. The substrate-level contribution the project has built — a deterministic Hypernetwork with FiLM-restricted output at three injection sites of a recurrent PPO policy, upgradable to a finite-mixture approximation of a Bayesian Hypernetwork via the FiLM-Ensemble + heteroscedastic-precision compound (Row EE-6 in the concept memo) — is a stronger and more honestly-framed contribution if it is **decoupled from any specific channel-to-site mapping** and pitched as **one demonstration that a single side-signal can drive heterogeneous behavioural readouts at multiple sites of a forward-pass FiLM substrate**. The Doya channels become *interpretive frames* applied to the dissociation profile post-hoc, not architectural commitments built in upfront. This is **Rung 3** in the user's framing — softer than v4 — and I recommend the project move to it.

On the impactful-vs-reasonable trade-off, I recommend the **middle position**: a NeurIPS / TMLR-shaped 8-10 page paper with Row EE-6 (FiLM-Ensemble + heteroscedastic-precision compound) as the **architectural headline**, the **R2 anchor** (the project's existing empirical win, where the modulated agent recovered roughly 25× the seed-noise floor faster than the unmodulated baseline on a second return to a previously-seen task stage) reframed as a **substrate-level demonstration of behavioural-effect heterogeneity from a single side-signal**, and the Doya / Bayesian-brain readings positioned in **Discussion / Future Work** as interpretive lenses. v4's full four-arm clamp falsifier becomes one figure inside the paper; the per-site Doya-channel attribution is downgraded from claim to discussion question.

My two new ideas this round are: (E1) the **observation-point framing** — reframe sites A/B/C not as three channels carrying three named hyperparameters but as three *observation points* on the same modulator signal's effect, with the dissociation profile being a *behavioural fingerprint* rather than a channel decomposition; and (E2) the **Galanti-Wolf 2020 modularity bound** as the formal grounding for *why* one signal can produce dissociable readouts without being decomposed — a result the v4 memo cites for capacity but not for identifiability of one-to-many readouts. A third candidate idea, (E3), proposes the **hypernet lineage itself as the paper** — pitching the project's contribution as evidence that a finite-mixture approximate Bayesian Hypernetwork (Row EE-6) produces behavioural-effect heterogeneity on an RL policy, with neuromodulation as the motivating biological framing rather than the mapping target. I sketch all three below.

The v4 memo is **not architecturally broken** — its math is clean, its substrate boundary is honest, and its narrowing from v3's four-channel claim to a two-channel-cleanly-carried claim is a real improvement. But under the user's added "possibility + one-to-many" constraint, v4 still inherits Doya 2002's one-to-one channel attribution at sites A and C ("α at A, β at C") which the user has explicitly named as too strong. The recommendation is to keep v4's substrate and experiments and **soften the claim a third time**, not to redesign anything.

---

## §A The possibility-bridge rungs — architectural commitment ladder

The user's "possibility, not full coverage" framing maps onto an architectural commitment ladder. Each rung asks the same question — *what does the paper claim the FiLM substrate is doing?* — at a different strength.

**Rung 1 (v3-era, strongest).** "The FiLM / hypernet family is the unifying framework for forward-pass neuromodulator-inspired RL algorithms; the project's substrate instantiates four Doya channels at three sites." This is v3's position. It collapsed under the math-reviewer audit (flag (d): site B is not γ_Bellman; flag (c): the Hessian discriminator direction was reversed; flag (b): the EE-6 compound formula needed two design choices closed).

**Rung 2 (v4, current).** "FiLM at three sites cleanly carries two of Doya's four channels — α at site A (encoder) and β at site C (policy logits); site B is re-named plasticity gating; γ_Bellman is acknowledged as out-of-substrate." Much more honest after the audit. But it still commits to a *channel-to-site* mapping (α-at-A, β-at-C) that the user's no-one-to-one constraint flags as too strong. A neuroscience reviewer will (correctly) note that ACh projects widely (basal forebrain → cortex, brainstem → many targets); restricting ACh-α to the encoder is a *modelling choice* the v4 paper presents as a *biological mapping*.

**Rung 3 (recommended).** "A single learnable side-signal, fed through a FiLM substrate at three injection sites, produces dissociable behavioural readouts — sufficient to instantiate *multiple* Doya-style behavioural-hyperparameter effects from one modulator. We make no claim that site A IS the ACh site or site C IS the NA site; the dissociation profile across A, B, C is *consistent with* the kinds of multi-effect-from-one-signal mappings the neuromod literature describes biologically. This is a *substrate-level demonstration of one-to-many modulation*, not a *channel-decomposition mapping*." Every v4 experiment is preserved; only the interpretation softens. The paper's headline becomes the *capability*, not the *correspondence*.

**Why Rung 3 is the right rung for the project's current empirical state.** The only anchor the project has today is the R2 win — one behavioural readout in one task. Even v4's four-arm clamp at site C is not yet run. Claiming the Doya channel-to-site mapping on this evidence is a forward bet; claiming a single-signal-many-effects demonstration is a backward-compatible reading of what the architecture has already shown. Under Rung 3, the heterogeneity arises from the three site-specific read-out matrices, not from the modulator emitting separable channels — exactly the architectural pattern Galanti & Wolf 2020 Theorem 4 gives the formal grounding for (see §E2). One signal, many effects, by architectural construction.

---

## §B No-one-to-one — architectural consequences

The user has named the no-one-to-one constraint twice in this round of work. It has architectural consequences the project needs to absorb explicitly.

**Doya 2002's one-to-one mapping is a didactic simplification, not a biological fact.** ACh modulates learning rate, attention, signal-to-noise, cortical-state switching, and cholinergic-interneuron-mediated DA modulation of striatum (Hasselmo, Sarter). NA modulates inverse temperature, gain, working-memory stability, large-scale brain-state switching (Aston-Jones & Cohen 2005), and per-region differential effects via α1/α2/β receptor heterogeneity. Lee 2024 inherited Doya's one-to-one structure and the cost is visible in their §6 limitations: no normative theory for the 5-HT-γ branch, dropped from the agent.

**v4 still partially commits to Doya's one-to-one structure.** v4 §3 reads: "Site A (encoder pre-fusion) — α channel. ACh-as-learning-rate ... Per-feature gain on the encoder's pre-fusion activations is the activation-level analogue of a per-feature learning-rate adjustment." This has the form "Site A IS the ACh-α site". Under the user's no-one-to-one constraint, this should be: "Site A *produces effects consistent with* α-style learning-rate modulation; we do not claim it is the ACh site, only that the dissociation profile at A is the *kind of profile* an ACh-like signal would be expected to produce on this substrate."

**The architectural pattern under no-one-to-one is "one signal, heterogeneous read-outs" — and is architecturally native to what the project has already built.** Galanti & Wolf 2020 Theorem 4: a hypernetwork's primary network produces parameter-efficient context-conditional weights *because* the read-out matrices carry the per-target specificity, while the conditioning signal is shared. The project's three-site FiLM substrate is a hypernetwork (per [investigation memo Claim 1](../../concepts/film_hypernet_bnn_lineage_and_critic_modulation.md)). One conditioner, many targets — by Theorem 4, this is *the* architectural pattern hypernetworks exist for. The user's no-one-to-one framing is *architecturally native* to what the project has already built — just not yet rhetorically claimed.

**Row EE-6 is the cleanest no-one-to-one anchor.** The FiLM-Ensemble + heteroscedastic-precision compound ([concept memo §5.4 Row EE-6](../../concepts/film_neuromod_integration.md), under [investigation memo Claim 4](../../concepts/film_hypernet_bnn_lineage_and_critic_modulation.md)) is a finite-mixture approximation of a Bayesian Hypernetwork. Its $M$ ensemble members produce $M$ deterministic settings of $(\gamma^m, \beta^m)$ that are NOT a channel decomposition — they are *samples from a posterior over the read-out*. "Many effects" emerge from posterior variance, not from channel attribution. The cleanest "one signal, many effects via posterior over read-outs" framing in the project's inventory.

**Implication for v4 is surgical, not invasive.** v4's substrate stays. v4's experiments stay. What changes is the rhetorical positioning of sites A and C — from "ACh / NA mapping" to "observation points on a single modulator signal whose dissociation profile is consistent with multi-effect modulation". The Doya channels become *interpretive lenses* the Discussion applies post-hoc, not the architectural target the substrate was built to instantiate.

---

## §C Three alternative paper framings

Three framings, ordered by increasing distance from v4. All preserve v4's substrate and experiments; what changes is rhetorical scope.

**Framing C1 — Substrate-level demonstration of multi-effect modulation.** Central claim: *one learnable side-signal produces dissociable behavioural readouts at multiple sites of a forward-pass FiLM substrate, with the heterogeneity arising from per-site read-out matrices rather than channel decomposition*. Architectural contribution: the FiLM-Ensemble + heteroscedastic-precision compound (Row EE-6), framed as the first approximate Bayesian Hypernetwork with heteroscedastic likelihood for an RL policy. Empirical contribution: the R2 anchor + v4 §5.1 four-arm clamp at site C. Interpretive contribution (Doya / Lee / Vecoven) → Discussion. Reader takeaway: "the bridge between neural gain and behavioural-hyperparameter modulation can be built; here is one instance".

**Framing C2 — Row EE-6 as the architectural headline (recommended).** Central claim: *the first approximate Bayesian Hypernetwork with heteroscedastic likelihood for an RL policy*. The neuromodulator framing is motivation, not target. The substrate is positioned in the FiLM ⊂ Hypernet ⊂ point-mass-BHN-limit family ([investigation memo §2](../../concepts/film_hypernet_bnn_lineage_and_critic_modulation.md)); Row EE-6 is a substrate move *down* toward stochastic context-conditional weights. Neuro lit → motivation; DL lit → formal home; empirical anchor → R2 + dissociation across A, B, C. Maximises architectural-contribution-per-page ratio. NeurIPS / ICLR architecture-track.

**Framing C3 — Bare R2 win reframed (lowest commitment).** Central claim: *target-flexible gain modulation produces multiple behavioural-hyperparameter-like effects under regime change*. No Doya channel-to-site commitment, no Row EE-6. Workshop-shaped; under-uses Row EE-6, which is the project's strongest architectural contribution.

**Cross-framing observation.** C1 reads as a substrate-level paper with biological motivation; C2 reads as a Bayesian-DL paper with neuro motivation; C3 reads as an RL paper with a substrate-level observation. Architectural cost of moving between them is zero (no code changes); the cost is reviewer-positioning.

---

## §D Position on the impactful-vs-reasonable trade-off

Three positions exist; my recommendation is the middle.

**Most impactful (Framing C1 + full predictions program).** Row EE-6 as architectural headline + the v4 channel-attribution claim + the P-1 through P-10 predictions ([v3 sidecar](../../references/neuromodulatory_algorithms/feedback/20260516_rl_bayesian_dl_predictions_for_v3.md)) + the T/P split prerequisite + the C1 EMA baseline contrast + the probe-trial extension + cross-context transfer. Multi-month timeline. **Risks**: non-trivial experimental extensions not yet built; channel-to-site attribution still rhetorically too strong under the no-one-to-one constraint; Row EE-6 implementation not yet running. **Verdict**: too much commitment for the current evidence base.

**Most reasonable (Framing C3 + R2-only).** Workshop submission. Single framing ("one FiLM modulator → dissociable readouts at three sites"); R2 anchor is the only result needed; no T/P split, no probe trials, no Row EE-6. **Verdict**: under-uses the project's strongest architectural contribution; too soft for a full-paper headline.

**Middle (recommended) — Framing C2 + the four-arm clamp at site C + the 3×3 dissociation matrix + Row EE-6 as the substrate upgrade.** The paper claims: (a) substrate is positioned in the FiLM ⊂ Hypernet family per the lineage investigation; (b) Row EE-6 is the architectural contribution — first approximate Bayesian Hypernetwork with heteroscedastic likelihood for an RL policy; (c) empirical anchor (R2 win + four-arm clamp + 3×3 matrix) demonstrates single-signal-many-effects dissociation; (d) Doya / Bayesian-brain readings live in Discussion. **Architectural cost**: implement Row EE-6 (non-invasive — a new head); four-arm clamp harness (already easy under per-site freeze flags); 3×3 dissociation evaluation; T/P split as optional Phase 2 add. NeurIPS / TMLR-shaped 8–10 pages. **Verdict**: this is the unique point on the trade-off curve that fits the current evidence base, the v4 audit-corrected content, and the user's two new constraints. Pushing higher requires experiments not yet built; retreating wastes audit work that already exists.

---

## §E NEW ideas this round

Three ideas I have not raised in v1 §8, the v3 sidecar, or the lineage investigation memo. They are paper-shape proposals, not experiment-design proposals.

### §E1 — The "observation-point" reframing of sites A, B, C

**Idea.** Drop the "channel-to-site" mapping (α-at-A, β-at-C) from v4. Replace with: sites A, B, C are three *observation points* on the effect of a single modulator signal. The dissociation profile across A, B, C is the *behavioural fingerprint* of the signal acting on heterogeneous targets via three different read-out matrices. *Not* a decomposition into separable channels.

**Fits both constraints.** The "possibility" framing follows: A, B, C *demonstrate* dissociable readouts from one signal, they don't *implement* Doya's α, β, γ. The "one-to-many" framing follows by construction: one modulator → three site-specific effects via Galanti & Wolf 2020 Theorem 4's per-target read-out modularity, not via the modulator emitting three independent values.

**Architectural cost: zero.** Substrate unchanged; experiments unchanged. v4 §5.2 (α-channel at A) becomes "dissociation profile at site A, interpreted as evidence for α-like effects". §5.3 (β-channel at C) becomes "profile at C, interpreted as evidence for β-like effects". §5.4 (plasticity-gating at B) is already in this form per the math-reviewer audit.

**Precedent.** Galanti & Wolf 2020 Theorem 4 — one-conditioner-many-per-target-readouts is *the* architectural form for parameter-efficient context-conditional weights. Costacurta 2024 Fig. 3F: dissociable-by-ablation in NM-RNN shown as a *capability*, not a channel-to-function mapping. Vecoven 2020 Fig. 7: $z$-recruitment as diagnostic, not attribution. Both precedents show dissociation rhetorically the way E1 proposes.

### §E2 — Galanti-Wolf Theorem 4 as the formal grounding for one-to-many readouts

**Idea.** Cite Galanti & Wolf 2020 Theorem 4 as the *identifiability* anchor for why one signal can produce dissociable readouts without being decomposed into separable channels. v4 currently cites the theorem only for *capacity* ([concept memo §3.0 Claim 5](../../concepts/film_neuromod_integration.md); [v4 §3.0 Claim 5](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md)): the parameter-count bound $N_g = O(\epsilon^{-m_1/r})$ vs embedding-concatenation $\Omega(\epsilon^{-(m_1 + m_2)/r})$. This is the efficiency reading. There is a second reading more load-bearing for the no-one-to-one framing.

**The identifiability reading.** Theorem 4's parameter-efficiency comes from read-out matrices carrying the per-target specificity while the conditioner is shared. The modulator signal does *not* need to be decomposed into per-target channels for the architecture to produce dissociable per-target effects. The dissociation lives in the read-out, not in the signal. A reviewer asking "if one signal is producing dissociable effects at three sites, what *are* the three effects in the signal itself?" has the formal answer: nothing — the signal is low-dimensional; the effects live in the read-outs; Galanti-Wolf Theorem 4 says this is how hypernetworks work.

**Paper structure.** Move the Galanti-Wolf citation from Claim 5 (capacity) to a separate paragraph titled "Architectural grounding for one-to-many readouts". One-paragraph addition; works under any framing C1/C2/C3.

### §E3 — The hypernet lineage as the paper itself

**Idea.** Pitch the project's contribution as evidence that the FiLM ⊂ Hypernet ⊂ point-mass-BHN-limit family, with Row EE-6 as a finite-mixture approximate Bayesian Hypernetwork, produces behavioural-effect heterogeneity in RL — a *pure architectural / lineage paper*. The neuromodulation framing becomes a small motivation paragraph; the paper's bulk is the architecture, lineage, and empirical demonstration.

**Why this is interesting.** The investigation memo's five formal claims ([§2](../../concepts/film_hypernet_bnn_lineage_and_critic_modulation.md)) — FiLM ⊂ Hypernet, Hypernet ⊆ point-mass-BHN-limit, BNN ⊥ BHN, FiLM-Ensemble ≈ finite-mixture-BHN, Galanti-Wolf modularity carries through — are each individually known, but their *combination* (explicit 2×2 placement + Row EE-6 as the move down to stochastic context-conditional weights) has not appeared as a single paper. Audience: the Bayesian-DL / approximate-inference community (Krueger 2017, Kendall & Gal 2017, Turkoglu 2022). Venue: NeurIPS BDL workshop, AABI, possibly ICLR.

**Caveat.** Partially abandons the neural-gain-meets-hyperparameter-modulation framing the user has invested four direction memos into. I include E3 as the high-impact-architectural option but recommend C2 instead — which keeps neuro as motivation while still letting Row EE-6 carry the headline.

---

## Cross-references and sign-off

### What I am NOT recommending

I am not recommending any code-side or experiment-design change in this contribution. The user has explicitly named the project as "still in planning". My contribution is at the level of paper framing and rhetorical positioning.

I am not recommending v4 be discarded. v4 is technically sound, audit-corrected, and a real improvement over v3. The recommendation is to revise v4's claims one more time (from Rung 2 to Rung 3 per §A above) and reframe sites A, B, C as observation points rather than channels per §E1. This is surgical, not invasive.

I am not recommending Row EE-6 be implemented before the paper is scoped. The implementation cost of Row EE-6 is non-trivial (FiLM-Ensemble at the actor head + heteroscedastic precision head + the ρ-init choice + the variance-space disambiguation). The paper scope should be decided first; the architectural-headline status of Row EE-6 in Framing C2 makes it worth implementing if and only if C2 is the chosen framing.

### Key flag for v4's architectural defensibility under the no-one-to-one constraint

**v4 is not architecturally problematic.** v4's substrate (deterministic Hypernetwork with FiLM-restricted output at three sites) is *natively one-to-many* — by Galanti & Wolf 2020 Theorem 4, that is what the architecture exists to support. v4's experiments (four-arm clamp at C, 3×3 dissociation matrix, T/P split) are *all compatible* with a Rung 3 / observation-point framing. The only thing problematic under the user's no-one-to-one constraint is v4's *rhetorical* commitment to a channel-to-site mapping (α-at-A, β-at-C). That commitment is removable without touching any code, any experiment, or any equation.

**Recommendation: a v5 of the direction memo** that softens v4's channel-to-site mapping to the observation-point framing of §E1, cites Galanti-Wolf Theorem 4's identifiability implications per §E2, and aligns with Framing C2 (Row EE-6 as architectural headline + R2 as motivating empirical anchor + four-arm clamp at C as falsifier + Discussion frames the dissociation profile in Doya / Bayesian-brain terms post-hoc). No code changes. No new experiments beyond Row EE-6 implementation.

### Cross-references

- **Inputs read this round**: [v4 direction memo](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md); [concept memo §3.0 + §5 + §10](../../concepts/film_neuromod_integration.md); [lineage investigation memo](../../concepts/film_hypernet_bnn_lineage_and_critic_modulation.md).
- **My prior contributions** (not re-stated in this memo): [v1 §8 architectural concerns](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation.md) — the original GRU-vs-Bellman-γ conflation flag (later validated by math-reviewer flag (d) at v8); [v3 round sidecar P-1 through P-10](../../references/neuromodulatory_algorithms/feedback/20260516_rl_bayesian_dl_predictions_for_v3.md) — the ten architectural predictions; [lineage investigation memo](../../concepts/film_hypernet_bnn_lineage_and_critic_modulation.md) — the FiLM ⊂ Hypernet ⊂ point-mass-BHN-limit derivation, the 2×2 family placement, the γ_Bellman-as-loss-not-forward-pass result.
- **Closest published precedents for the Rung 3 / observation-point framing**: Costacurta 2024 Fig. 3F (dissociation-by-ablation as a capability, not a channel mapping); Vecoven 2020 Fig. 7 ($z$-recruitment as a diagnostic, not an attribution); Galanti & Wolf 2020 Theorem 4 (the formal grounding for one signal → heterogeneous readouts via per-target read-out matrices).
- **Closest published precedents for Framing C2 / Row EE-6 as architectural headline**: Krueger 2017 (full BHN, no heteroscedastic head, not RL); Kendall & Gal 2017 (BNN, no context-conditioning, not RL); Turkoglu 2022 (FiLM-Ensemble, vision/supervised, not RL). The Row EE-6 compound's novelty is the combination of all three on an RL policy.

---

*Contribution by `professor-rl-bayesian-dl`, 2026-05-16.*
