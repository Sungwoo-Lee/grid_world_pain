---
title: "Doya modulation via FiLM/hypernet — concise summary"
status: draft
author: top-level-claude
audience: user + pi + research-postdoc
date: 2026-05-21
summary_of: "docs/project/ideas/20260519_doya_modulation_extended_creative_ideas_v2.md"
companions:
  - "docs/project/ideas/20260518_doya_modulation_via_film_hypernet_implementation_ideas.md"
  - "docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md"
one_line_summary: "Executive summary of the v2 extended ideas memo — the two-headed architecture, a per-knob candidate table, the recommended first prototype per knob, and the cross-cutting insight that the richest design per knob outputs a vector or function from Head 2 rather than a scalar."
---

## TL;DR

A single **modulator** network with **two output heads** can implement all four neuromodulator-to-RL-hyperparameter assignments Doya (2002) proposed — noradrenaline ↔ policy temperature, acetylcholine ↔ learning rate, dopamine ↔ reward-prediction-error gain, serotonin ↔ time-discount. The architecture is the same across all four knobs; the *richest* candidate per knob outputs a **vector** or **function** from Head 2 rather than a scalar.

## The two-headed architecture

```
                  context c (interoceptive state)
                              |
                       [ modulator m_φ ]
                         /          \
                  Head 1              Head 2
          (FiLM γ, β) → activations  (scalars / vectors / functions) → loss / return
                  |                       |
              forward pass            backward / Bellman
```

- **Head 1** modifies the agent's *forward pass* by scaling and shifting hidden activations (FiLM). By chain rule it also reaches the backward pass — so Head 1 alone gives per-pathway learning-rate gating "for free."
- **Head 2** modifies the *loss* and *return estimator* — what gets multiplied into the TD error, what discount is used in the Bellman target, what the entropy bonus coefficient is.

## Per-knob candidate table

| Knob | Candidate 1 (simplest) | Candidate 2 | Candidate 3 (richest) |
|---|---|---|---|
| **NA — temperature** | Rank-1 broadcast FiLM γ at policy logits ≡ tempered softmax | Mellowmax-as-FiLM-target — per-state ω chosen to enforce a non-expansion constraint (Asadi & Littman ICML 2017) | Munchausen-style log-policy bonus with FiLM-conditioned coefficient (Vieillard NeurIPS 2020) |
| **ACh — learning rate** | FiLM γ as a per-pathway chain-rule gradient gate (Highway NeurIPS 2015, GLU ICML 2017, MoE ICLR 2017) | Hypernet emits per-context weights; effective LR via Jacobian pullback (Ha ICLR 2017, Sarafian ICML 2021) | Meta-gradient Head 2 emits global α(c) (Xu NeurIPS 2018, IDBD AAAI 1992) |
| **DA — TD-error gain** | Loss-side scalar κ(c) multiplies the TD error (the audit's baseline) | **Successor-features Head 2 emits reward weight vector w_r(c) ∈ ℝ^d** (Barreto NeurIPS 2017, Borsa ICLR 2019) | IQN-style distortion preference β(c) on the return distribution (Dabney ICML 2018) |
| **5-HT — discount** | γ-conditional UVFA — Head 2 emits γ_B(c) into the TD target (Sherstan AAAI 2020, Schaul ICML 2015) | Multi-horizon ensemble — Head 2 emits the discount *shape* w(γ; c), exponential ↔ hyperbolic (Fedus 2019, Romoff ICML 2019) | γ-Models — context-conditioned generative dynamics; modulator sets the imagination horizon (Janner NeurIPS 2020) |

## Recommended first prototype per knob

**NA — Candidate 2 (Mellowmax-anchored).** The cleanest theoretical defence against CAT-SAC-style rejection (ICLR 2021): mellowmax's per-state ω comes from a *convergence theorem*, not from intuition. Wire the modulator to emit the mellowmax target ω(c); critic and policy use mellowmax in place of softmax. *Why first:* anchors the most-competed knob to top-venue theory.

**ACh — Candidate 1 (FiLM chain-rule channel).** The audit's chain-rule reading makes this nearly free: any FiLM γ already in the forward pass for NA *is* a per-pathway α-gate in the backward pass. *Why first:* requires zero new architecture beyond what NA already provides — surfacing the existing mechanism with a stratified-direction empirical test (the audit's signature) is the contribution, not a new module.

**DA — Candidate 2 (Successor-features w_r(c)).** The single highest-value structural upgrade in the whole memo. Replacing scalar κ(c) with a vector w_r(c) ∈ ℝ^d **breaks the DA/ACh identifiability gauge** that the audit flagged, and connects DA modulation to the well-developed successor-features literature. *Why first:* it is the upgrade that actually changes what the project can *claim* about DA.

**5-HT — Candidate 1 first, then escalate.** Start with the γ-conditional UVFA (Head 2 emits scalar γ_B(c)); validate against the calibrated-effective-horizon test; then escalate to multi-horizon ensemble (Candidate 2) or γ-Models (Candidate 3) only if the scalar route shows promise. *Why this ordering:* the cheaper-first ablation isolates whether the discount-as-context-output works at all before adding architectural complexity.

## Cross-cutting insight

**The richest candidate per knob outputs a *vector* or *function* from Head 2, not a scalar.** Successor features (DA) emit a vector w_r(c); multi-horizon ensembles (5-HT) emit a weighting function w(γ; c); hypernet-emitted critic weights (DA, ACh) emit a tensor θ(c); γ-Models (5-HT) emit a full conditional dynamics model. The audit's two-headed framing was right *architecturally* but understated the *dimensionality* of Head 2's output. The single most consequential update to the project's framing: **Head 2 is not just a scalar emitter; it is a structured-output emitter**, and the dimensionality is what breaks the audit's identifiability gauges and unlocks construct-validity claims.

**All four knobs can share a single modulator trunk.** The NMN unification dream made architecturally honest: one shared encoder of context c, four output heads (one per knob), each at the right dimensionality for its knob.

## Honesty constraints that survive

1. **DA/ACh gauge.** Scalar Head 2 outputs for both are not separately identifiable from on-policy training. Vector / structured outputs break this. (Resolved by Candidate 2 for DA.)
2. **ACh coupling.** FiLM γ in the forward pass cannot modulate learning rate *without* simultaneously feature-gating. The audit's chain-rule addendum is honest about this; the v2 memo carries it forward.
3. **5-HT well-definedness.** A context-conditioned discount γ_B(c) requires either c constant within trajectory or context-absorbed-into-state framing — bookkeeping, not blocker.
4. **NA reward-bonus baseline.** The CAT-SAC rejection (ICLR 2021) flagged that "modulator output → temperature" is empirically indistinguishable from "modulator output → intrinsic reward bonus" unless the architectural distinction is derived from a principle. Our NA Candidates 2 and 3 (Mellowmax, Munchausen) inherit the theoretical foundation; Candidate 1 (basic FiLM-γ) does not on its own.

## Pointers

- **Full v2 extended ideas memo (the source this summarises)**: [`docs/project/ideas/20260519_doya_modulation_extended_creative_ideas_v2.md`](20260519_doya_modulation_extended_creative_ideas_v2.md) — 12,000 words, 8 sections, 2–3 candidates per knob with plain-language algorithm intros, references, and "what I'd try first" per knob.
- **V1 compact memo**: [`docs/project/ideas/20260518_doya_modulation_via_film_hypernet_implementation_ideas.md`](20260518_doya_modulation_via_film_hypernet_implementation_ideas.md) — 2,400-word architecturally-honest summary, includes per-paper γ-Nets / UVFA / Fedus integration.
- **Theoretical audit (the anchor)**: [`docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md`](../critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md) — original three-round audit (initial verdict, ACh chain-rule addendum, 5-HT γ-conditional-UVFA addendum).
- **Audit story doc (reader-facing)**: [`docs/project/critiques/20260518_film_as_hyperparameter_modulator_audit_story.md`](../critiques/20260518_film_as_hyperparameter_modulator_audit_story.md) — plain-English walkthrough of the audit arc.
- **Lit-review folders**: `docs/project/references/{Temperature, Learning_Rate, TD, Gamma}/` — four review files per topic plus source PDFs in `sources/`.
- **V5 direction memo (predates this work; not yet superseded)**: `docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v5.md` — the project's framing-decision document. A v6 that integrates the audit, the v1/v2 ideas memos, the CAT-SAC diagnostic, and the four lit-review corpora has not been drafted.
