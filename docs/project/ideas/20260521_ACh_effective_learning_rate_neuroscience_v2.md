---
title: "ACh and the effective learning rate — affective + computational neuroscience perspective (v2 stage)"
status: draft
author: top-level-claude (synthesis of user-Claude discussion, 2026-05-21)
audience: user + pi + research-postdoc
date: 2026-05-21
v_stage: v2
companions:
  - "docs/project/ideas/20260519_doya_modulation_extended_creative_ideas_v2.md"
  - "docs/project/ideas/20260521_doya_modulation_summary_v2.md"
  - "docs/project/ideas/20260521_5HT_effective_discount_neuroscience_v2.md"
  - "docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md"
integration_status: "Standalone perspective memo, NOT yet integrated into the v2 extended ideas memo, the audit, or the lit-review folders. The user will integrate later. Parallel to the 5-HT and NA effective-perspective memos."
one_line_summary: "Reframes ACh's role from 'modulating the learning-rate parameter α' to 'gating expected-uncertainty precision on bottom-up signals' — better-grounded in affective and computational neuroscience and matching the FiLM-chain-rule architectural route the audit already established."
---

## 1. What this doc is

A standalone perspective memo on the **ACh → learning-rate** knob of the project's neuromodulator-network framing, parallel in structure to the 5-HT effective-discount memo (`20260521_5HT_effective_discount_neuroscience_v2.md`). It is **not** an integration into the v2 extended ideas memo, the audit, or the lit-review folders. It captures a single reframing: the *effective learning rate* reading of ACh is more biologically realistic than the *implementation learning rate* reading the project has been using, and the architectural consequences strongly favour the FiLM-chain-rule route the audit already identified.

A reader with basic RL only: in textbook gradient descent, the learning rate $\alpha$ is the scalar that multiplies the gradient in the parameter update $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$. Doya (2002) mapped this $\alpha$ onto the cholinergic (ACh) system. **This memo argues that the mapping is closer to the truth if we drop "scalar parameter" and replace it with "expected-uncertainty-gated bottom-up precision."** The brain does not have an $\alpha$; it has receptor-specific gain modulation of bottom-up sensory and prediction-error signals, and a separate set of plasticity-gating mechanisms (NMDA, BDNF, synaptic tagging). What looks like "the brain set its learning rate" is an emergent consequence of how ACh gates which signals propagate strongly and which synapses are eligible to change.

## 2. The shift in framing — implementation α versus effective learning rate

The project's v2 extended memo handles ACh under the chain-rule reading from the audit's 2026-05-18 addendum: a FiLM $\gamma$ in the forward pass is also a diagonal Jacobian re-weighting in the backward pass, so per-pathway *effective* learning rates are gated by the FiLM scale. This was already the right architectural reading. **What this memo adds is the neuroscience reading that makes it the *right* reading, not merely a *clever* one.**

The implementation-α reading says the modulator emits a number that gets multiplied into a gradient update. The brain does not do this. There is no neural locus where a scalar gets read off and multiplied into a plasticity rule. Instead, ACh modulates:

- **Bottom-up signal gain** in cortex via nicotinic and muscarinic receptors, with region-specific receptor distributions (e.g., M1 in cortex, α7-nAChR in attentional circuits).
- **Synaptic plasticity eligibility** — ACh enhances LTP and reduces depression in cortical and hippocampal circuits during specific behavioural states.
- **Encoding-vs-retrieval mode** in hippocampus — high ACh favours encoding of new information; low ACh favours retrieval.
- **Cortical map plasticity** — pairing ACh release (basal forebrain stimulation) with a sensory stimulus expands the cortical representation of that stimulus (Kilgard & Merzenich 1998).

Each of these is a forward-pass operation, with a downstream effect on what gets learned. **The effective learning rate is what we measure behaviourally; the actual mechanism is forward-pass gain on bottom-up signals plus plasticity-eligibility gating.** Both gating and gain are exactly what FiLM γ implements via the chain-rule channel the audit established.

The cost-vs-benefit trade-off is the same shape as the 5-HT case but milder. The project can directly *implement* per-pathway effective learning rates (this is what FiLM γ does, with the empirical signature from the audit's stratified-direction test). What it cannot do is read off a single scalar α and apply it globally; in the brain there is no such scalar. The cost is the inability to set a global α; the feature is that the brain does not have one either.

## 3. Affective neuroscience grounding

Four lines of evidence ground the effective-learning-rate reading in affective and behavioural neuroscience.

**Uncertainty-driven learning-rate shifts in associative learning.** A century of associative-learning research (Pearce-Hall 1980, Mackintosh 1975, more recent Behrens et al. 2007) shows that the *rate at which* humans and animals update beliefs is modulated by the volatility of the environment — when surprises happen often, learning accelerates; when the environment is stable, learning rates fall. Behrens et al. (2007) specifically showed that this volatility-tracking is fronto-parietal and modulated by uncertainty-sensitive neuromodulators. The classical Pearce-Hall model has *attention* as the modulator; later work identifies that attention as cholinergic. The brain's "learning rate" is thus an emergent property of how attention is allocated to predictively-relevant signals, not a parameter that gets adjusted directly.

**Hasselmo's encoding-retrieval framework.** Hasselmo & McGaughy (2004) and many follow-ups: ACh in the hippocampus modulates the balance between *encoding new information* and *retrieving stored information*. High ACh tilts toward encoding; low ACh toward retrieval. This dichotomy is conceptually equivalent to "high effective learning rate" vs. "low effective learning rate" — but the brain implements the switch by gating which inputs propagate (afferents during high ACh, recurrent during low ACh), not by changing an α parameter. The forward-pass gating produces the learning-rate-like behavioural effect.

**Critical periods and cortical map plasticity.** Bear & Singer (1986) and Kilgard & Merzenich (1998): cholinergic input from the nucleus basalis to cortex gates the plasticity of cortical sensory maps during critical periods and during pairing protocols in adults. The mechanism is gain modulation of NMDA-receptor-dependent LTP, plus modulation of inhibitory tone. Again: forward-pass operations that *induce* high effective learning rate when the cholinergic system fires alongside salient sensory input.

**Anticholinergic deficits and dementia.** Acute scopolamine (anticholinergic) administration impairs declarative memory formation; Alzheimer's disease — characterised in part by basal forebrain cholinergic degeneration — has effective learning rate collapse in cortical and hippocampal circuits. Cholinesterase inhibitors (donepezil) partially restore effective learning, by increasing ACh tone. These clinical observations only make sense under the effective-learning-rate reading; under the implementation-α reading, "ACh sets α and Alzheimer's lowers α" is a tautology that does not predict the pattern of behavioural deficits or the partial recovery with cholinesterase inhibition.

**Summary.** Affective and behavioural neuroscience consistently treats the brain's learning rate as a *measured* property of how attention and uncertainty are allocated, not as a parameter the brain reads off and applies.

## 4. Computational neuroscience — three implementations the brain actually has

Three concrete mechanisms by which the brain produces effective-learning-rate shifts without ever multiplying a global α into a gradient.

**(1) Expected-uncertainty signalling via ACh (Yu & Dayan 2005).** This is the canonical computational interpretation. Yu & Dayan formalised the joint role of ACh and NA in uncertainty processing: **ACh signals *expected* uncertainty** (familiar contexts where you know how noisy things are), while **NA signals *unexpected* uncertainty** (novel violations of the model). Under their framework, ACh-modulated precision on bottom-up signals controls how strongly current sensory evidence revises beliefs — high precision (high ACh in expected-uncertainty contexts) produces fast belief updates, which behaviourally looks like a high learning rate. There is no α anywhere in this framework; only a precision-gating computation that is mathematically equivalent to multiplying the gradient by a context-dependent scalar at each cortical site.

This is the *most directly relevant* computational-neuroscience interpretation for the project. The interoceptive modulator c is structurally analogous to the brain's representation of expected uncertainty; the FiLM γ output is structurally analogous to ACh-modulated cortical gain. The mapping is no longer "we built the ACh analogue;" it is "we instantiated the precision-gating computation Yu & Dayan identified."

**(2) Predictive-coding precision modulation.** Friston, Adams, Brown, Stephan: in hierarchical predictive coding, prediction errors at each cortical level are weighted by *precision* — the inverse variance of the predicted signal. Precision is modulated by neuromodulators (ACh prominently). High-precision prediction errors at a given level produce stronger downstream updates of higher-level representations, which is behaviourally indistinguishable from a high learning rate at that level. This is the same fundamental mechanism as Yu & Dayan's, expressed in the predictive-coding framework that has become dominant in computational psychiatry.

For the project, this is the framework under which the modulator's output could be derived rather than merely posited. If the interoceptive state c is itself a marker of expected uncertainty (which it is, in the somatic-marker / interoceptive-inference tradition), then the modulator's emission of FiLM γ to gate cortical gain is *derivable from* the predictive-coding framework, not just an analogy.

**(3) Eligibility traces and synaptic tagging.** A separate but complementary computational interpretation: ACh modulates the synaptic tagging that determines which synapses become eligible for plasticity following a behavioural outcome. Frey & Morris (1997), Redondo & Morris (2011), more recent work on neuromodulator-dependent eligibility windows. The "learning rate" of a particular synapse is gated by whether that synapse was tagged within an ACh-permissive window before the plasticity-driving event. This is computationally equivalent to an attention-gated eligibility trace in actor-critic learning (Sutton & Barto chapter 7) — and again, it is a forward-pass operation that *induces* effective learning rates rather than setting a parameter.

For the project this connects to the chain-rule channel in a different way: the FiLM γ in the forward pass not only gates the gradient via the diagonal Jacobian (the audit's reading), it also functionally defines which *pathways* are eligible for plasticity at each context. The two readings are mathematically the same operator viewed from two angles, exactly as the audit's addendum noted.

These three are not mutually exclusive; they describe the same neuromodulator system from different mathematical perspectives. Across all three, the discount-equivalent for learning rate — **a global scalar α that gets read off and applied** — does not exist in any of the implementations.

## 5. The empirical anchor — Yu & Dayan 2005

The cleanest single computational-neuroscience anchor for the effective-learning-rate reading is **Yu & Dayan (2005), *Uncertainty, Neuromodulation, and Attention* (Neuron 46:681–692)**. They formalised ACh as the carrier of *expected uncertainty* — the precision the brain attaches to its model of the current task — and NA as the carrier of *unexpected uncertainty* (model-violating surprises). Under this framework:

- ACh-modulated precision on bottom-up inputs controls how strongly each piece of evidence influences belief updating.
- The behavioural learning rate (Pearce-Hall α, Behrens et al. 2007's volatility-tracking) is a *consequence* of this precision modulation, not a parameter.
- The dissociation between ACh and NA in their model corresponds to the dissociation between learning-rate modulation and exploration modulation observed in pharmacology and lesion studies.

Yu & Dayan also explicitly note that the *implementation* of this in cortical circuitry is via gain modulation — precisely what FiLM γ implements. This paper is the bridge from Doya 2002's ACh-α mapping (which is essentially correct in spirit but wrong about mechanism) to the architecturally honest reading the project's FiLM-on-activations route implements.

The behavioural-evidence anchor is **Behrens, Woolrich, Walton & Rushworth (2007), *Learning the value of information in an uncertain world* (Nature Neuroscience 10:1214–1221)**, which directly demonstrated volatility-modulated learning rates in humans and tied them to fronto-parietal cortex — the cortical targets of basal-forebrain cholinergic projections.

## 6. Implications for the project's framing

Four implications, written here for the integration the user will do later.

**(a) The chain-rule channel is no longer a structural curiosity; it is the *correct* mechanism.** The audit's chain-rule addendum (2026-05-18, commit `7eb4dc0`) showed that FiLM γ in the forward pass is *also* a per-pathway effective-learning-rate gate. Under the effective-learning-rate reading, this is not an interesting algebraic coincidence — it is precisely how the brain implements ACh-style learning-rate modulation. The project's FiLM-modulator architecture *instantiates* the Yu & Dayan precision-gating mechanism rather than analogising to it.

**(b) The coupling caveat (chain-rule channel cannot modulate learning rate without simultaneously feature-gating) gains a biological defence.** The audit was honest that FiLM-as-ACh is *coupled* with feature gating — you cannot have the per-pathway α-gate without simultaneously gating which features the network attends to. Under the effective-learning-rate reading, **this is the correct biology** — ACh does not modulate learning rate independently of attention; the modulation is unified, exactly as Yu & Dayan's precision framework states. The "coupling caveat" stops being a caveat and starts being a faithfulness claim.

**(c) Construct validity strengthens.** Under the implementation-α reading, the project has to argue that the FiLM modulator "is" the ACh system because both modulate a learning-rate-like quantity. Under the effective-learning-rate reading, the project can argue that the FiLM modulator *implements* the same computational mechanism (precision/gain modulation of bottom-up signals) that ACh implements in the brain. The mapping shifts from analogy to instantiation.

**(d) The "Meta-Gradient RL emits α(c) globally" candidate (v2 memo's ACh Candidate 3) loses biological motivation.** Meta-gradient routes (Xu, van Hasselt & Silver 2018) emit a *global* scalar α(c) and update via differentiation through the inner loop. This is algorithmically clean but biologically implausible — there is no global α in the brain to emit, and no global meta-objective being differentiated. Under integration, Candidate 3 should probably be downgraded relative to Candidates 1 (chain-rule) and 2 (hypernet); Candidate 1 is now the *biologically correct* baseline rather than the simplest fallback.

## 7. Pointers and integration status

This is a **standalone perspective memo**. It does not modify the v2 extended ideas memo, the audit, the audit's story doc, the v1 compact memo, the v2 summary, or the lit-review folders.

- **V2 extended creative ideas memo (the doc this reframing is most directly relevant to)**: [`docs/project/ideas/20260519_doya_modulation_extended_creative_ideas_v2.md`](20260519_doya_modulation_extended_creative_ideas_v2.md) — Section 4 (ACh) is the section that would be reorganised under integration.
- **V2 summary memo**: [`docs/project/ideas/20260521_doya_modulation_summary_v2.md`](20260521_doya_modulation_summary_v2.md) — the ACh row of the per-knob candidate table is the part that would shift.
- **5-HT effective-discount perspective (sibling memo)**: [`docs/project/ideas/20260521_5HT_effective_discount_neuroscience_v2.md`](20260521_5HT_effective_discount_neuroscience_v2.md) — same structural argument applied to 5-HT.
- **Theoretical audit (with the ACh chain-rule addendum)**: [`docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md`](../critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md) — the chain-rule addendum already established the architectural reading; this memo establishes the biological reading.
- **Learning-Rate lit-review folder**: `docs/project/references/Learning_Rate/` — the four lit-review files there (L2O foundations, LR adaptation, meta-gradient RL, gated conditional architectures) ground the architectural side. The neuroscience anchors here (Yu & Dayan 2005, Behrens et al. 2007, Hasselmo & McGaughy 2004, Kilgard & Merzenich 1998, Frey & Morris 1997) are not in those reviews and would need to be added under integration.

**For the integration the user will do later:** the natural update is to add a one-paragraph neuroscience-anchor preface to the v2 memo's Section 4 (ACh), noting that the chain-rule reading from Candidate 1 *is* the biologically faithful mechanism (not merely an architectural option), and updating Candidate 3 (Meta-Gradient RL) to acknowledge its biological-implausibility cost. A Yu & Dayan 2005 anchor reference belongs in the Learning_Rate lit-review folder as a new file, or in a sibling neuroscience-anchors folder.
