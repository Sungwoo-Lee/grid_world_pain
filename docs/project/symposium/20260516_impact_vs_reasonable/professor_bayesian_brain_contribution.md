---
title: "Symposium contribution — Bayesian-brain/precision-coding position on impactful-vs-reasonable framing"
session: 2026-05-16_impact_vs_reasonable
contributor: professor-bayesian-brain
date: 2026-05-16
inputs:
  - docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md
  - docs/project/concepts/film_neuromod_integration.md
  - docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md
  - docs/project/concepts/active_inference_hypervigilance.md
prior_contributions:
  - docs/project/references/neuromodulatory_algorithms/feedback/20260516_bayesian_brain_predictions_for_v3.md (v3 round)
---

# Symposium contribution — Bayesian-brain/precision-coding position on impactful-vs-reasonable framing

## Plain-English entry point (read first)

The user has set two design constraints for the upcoming paper that, taken together, are unusually sharp. **First**, the paper should claim only the **possibility** of bridging two literatures — the "neural-gain" tradition in computational neuroscience (a single biological signal like noradrenaline or acetylcholine multiplies the steepness of many target neurons' input–output curves at once) and the "behavioural-hyperparameter" tradition in reinforcement-learning theory (one knob controls learning rate, another controls exploration, another controls the discount horizon, and so on). The user explicitly does not want the paper to claim a complete bridge, only that the project's substrate **demonstrates this bridge is achievable**. **Second**, the mapping between biological neuromodulators and behavioural functions **must not be one-to-one** — a single biological neuromodulator in the brain affects many downstream functions, and the user wants the paper to embrace that fact as a feature, not as a limitation to be apologised for.

This memo argues that these two constraints, far from being awkward to defend, are **exactly the structure that the Bayesian-brain / predictive-coding / active-inference family of theories was designed to capture**. In those theories a single quantity — a *precision* signal that weights how strongly the agent should listen to a particular information stream — naturally produces many downstream effects at once, because the same precision can be applied to a sensory stream (perceptual change), a transition stream (memory change), or a policy stream (action change). One signal, multiple effects, by construction. The Bayesian-brain framing therefore reads the two user constraints as **a request to lean on theory the project has already half-adopted**, not as a request to invent new ground.

The memo's recommendation: **frame the paper as a substrate-level demonstration that one signal produces channel-selective multi-target effects, sufficient to suggest — not prove — that the precision-modulation reading of biological neuromodulation is the right computational abstraction.** This claim is impactful (it is the first interoceptive-RL implementation of one-signal-many-precisions on a forward-pass FiLM substrate) and reasonable (it does not require committing the paper to active inference as a normative framework; that commitment is in the discussion). The memo proposes three concrete reframings the project could adopt, locates the recommendation on the impactful-vs-reasonable spectrum, states the one non-trivial Bayesian-brain impact claim that uniquely earns the impact half of the trade-off, and ends with three new ideas the symposium has not previously raised.

The math, the citations, and the falsifiers live in the sections below.

---

## §A — Why the Bayesian-brain lens is uniquely fit for the user's two constraints

The user's pair of constraints — "**possibility** of integration" and "**no one-to-one** mapping" — has a precise structural shape: the project wants a theoretical framework in which **one upstream signal naturally couples to multiple downstream functional effects**, expressed at a level of abstraction below specific neurotransmitter identities. The substrate must support a one-signal-many-effects reading without requiring the paper to commit, channel-by-channel, that ACh-is-this and NA-is-that.

The FiLM-as-Doya-channels reading the project's v4 direction memo currently leads with is **implicitly one-to-one**. In v4 §3, site A carries the ACh / learning-rate channel, site C carries the NA / inverse-temperature channel, site B carries the plasticity-gating channel — three sites, three channels, three biological labels. The math-reviewer audit ([film_neuromod_integration.md §10](../../concepts/film_neuromod_integration.md)) already pushed v4 to honestly drop the fourth (5-HT / discount) channel and re-name the third because the original one-to-one mapping did not survive. The taxonomy is honest, but its *structure* is still channel-decomposed: each FiLM site labelled with one biological identity and one RL hyperparameter.

The Bayesian-brain / precision-coding lens **starts from a different structural commitment**. In the predictive-coding generative model (Friston 2005; Rao & Ballard 1999; Bastos et al. 2012), the brain maintains a hierarchy of beliefs about hidden states $s_{1:L}$ producing observations $o$. At every level $\ell$, prediction errors $\varepsilon_\ell$ are weighted by a precision matrix $\Pi_\ell$:

$$
F = \sum_{\ell=1}^{L} \tfrac{1}{2}\,\varepsilon_\ell^\top \,\Pi_\ell\, \varepsilon_\ell \;-\; \tfrac{1}{2} \log |\Pi_\ell| \;+\; \text{const}.
$$

The precision matrices $\Pi_\ell$ are the *theoretical home* of neuromodulatory gain (Friston 2010; Kanai et al. 2015; Adams, Shipp & Friston 2013; Shine 2021 Box 3). And here is the structural fact that matters for the user's constraints: **the precision-weighting framework does not commit a single biological neuromodulator to a single precision level**. Yu & Dayan 2005 (*Neuron*, "Uncertainty, neuromodulation, and attention") explicitly maps ACh to expected uncertainty *and* NA to unexpected uncertainty, with the two systems jointly controlling *different combinations* of precisions over state vs. action. In Pezzulo, Rigoli & Friston 2015 (*Progress in Neurobiology*), the same precision $\gamma_G$ on expected free energy is described as having multiple downstream effects — pragmatic policy selection, epistemic information-seeking, exploration breadth — without commitment to a one-neuromodulator interpretation.

This is precisely **the structure the user has asked for**: one signal, many effects, no one-to-one mapping. The user's intuition is not idiosyncratic — it is the canonical Bayesian-brain stance.

Concretely, the project's three FiLM sites can be read as three precision read-outs from one upstream modulator state $h_{\text{mod}}$:

$$
\Pi_s \;=\; f_A(h_{\text{mod}}), \qquad
\Pi_z \;=\; f_B(h_{\text{mod}}), \qquad
\Pi_\pi \;=\; f_C(h_{\text{mod}}),
$$

with $\Pi_s$ (sensory precision — how much to trust observation) at injection A, $\Pi_z$ (state-transition precision — how much to trust the prior over hidden states) at injection B, and $\Pi_\pi$ (policy precision — how sharply to commit to one policy) at injection C. **One $h_{\text{mod}}$, three read-outs, three different effects** — the architecture *is* the one-signal-many-precisions abstraction, by construction. The biological-neuromodulator labels become *interpretive overlays* on top of this substrate, not its skeleton. This is the v4 memo's §3 "level-specific precision reading" promoted from a sidebar to a primary framing.

In the active-inference variant, the same logic holds for *one precision on expected free energy*. The agent selects policies via

$$
\pi^* \;\propto\; \exp\!\bigl(-\gamma_G \cdot G(\pi)\bigr), \quad
G(\pi) \;=\; \underbrace{\mathbb{E}_q\!\left[\ln q(s\mid\pi) - \ln p(s,o\mid\pi)\right]}_{\text{expected free energy}}
$$

with $G(\pi)$ decomposing into an epistemic term (information gain about hidden states) and a pragmatic term (expected reward under the prior preference $p(o\mid C)$). A single modulator-conditioned $\gamma_G$ shifts the *balance* of those terms — exploration-vs-exploitation, information-vs-reward — without the architecture having to decide which sub-term it controls. **One signal, three or four behavioural effects, no commitment to a channel decomposition.** This is the active-inference analogue of the user's "no one-to-one mapping" constraint, and it sits naturally on top of the project's substrate without further architectural change.

The FiLM-as-channels reading and the precision-coding reading **predict the same data when one knob is doing the work**, but they predict differently — and the project's data will tell them apart — when the architecture is doing genuine multi-channel work (v4 §3 already concedes this, §5.6.1 already proposes the effective-rank-of-γ probe). The key point for the symposium: **adopting the precision-coding framing as the lens does not require any new experiments**; it requires choosing how to write the paper.

---

## §B — Three concrete Bayesian-brain reframings for the paper

I propose three reframings, each compatible with the project's v4 substrate as-built, ordered from least to most theoretical commitment.

### Framing B1 — Precision-weighted FiLM (sharpened from v3-round)

**Tagline.** "FiLM as a substrate for precision-weighting in a hierarchical generative model — one modulator state, three precision read-outs, heterogeneous behavioural effects."

**Theoretical commitment.** Light. Adopts the hierarchical-predictive-coding generative-model language from Bastos et al. 2012 + Friston 2010 + Adams, Shipp & Friston 2013, *without* committing to active inference's expected-free-energy decomposition. The paper says: "the agent maintains a hierarchical posterior over hidden states; the modulator's three FiLM read-outs implement the precision-weighting that this posterior requires; multi-target heterogeneous effects arise because the three precisions sit at different hierarchical levels."

**What the paper claims.** That a single biologically-inspired modulator can implement Bayesian-brain *precision-weighting* at three levels of a hierarchical inference machine, and that the resulting behavioural effects (regime-change recovery, channel-selective hypervigilance, plasticity gating) are the **expected downstream consequences of single-signal precision-modulation**, not separate engineering knobs. Concretely: the v4 §5.6.1 effective-rank-of-γ probe and the per-site partial-regression probe (v3-round prediction P3) are the operational signatures of the level-specific-precision reading.

**Lineage anchors.** Friston 2010 (free-energy principle); Adams, Shipp & Friston 2013 ("Predictions not commands"); Kanai et al. 2015 (precision-as-attention); Shine 2021 Box 3 (cellular gain ↔ precision); Yu & Dayan 2005 (the original two-uncertainty / two-neuromodulator framework — orthogonal to the project's choice of Doya-α but well-positioned in the discussion).

**Why this satisfies the user's constraints.** "Possibility" framing: the paper *suggests* precision-weighting as the unifying abstraction without claiming the bridge is closed. "No one-to-one mapping": the three sites are three precision levels, not three neurotransmitter labels — biology gets to give one signal multiple effects because the substrate's read-out matrix is what produces the per-site dissociation.

### Framing B2 — Active-inference FiLM (more theoretical commitment)

**Tagline.** "FiLM substrate as an implementation of context-dependent expected-free-energy weighting — the modulator shifts epistemic-vs-pragmatic balance without committing to a channel decomposition."

**Theoretical commitment.** Heavier. Adopts active inference as the normative framework. The paper says: "the agent selects policies by minimising expected free energy; the modulator's role is to weight the EFE terms (epistemic, pragmatic, risk) according to context; the FiLM substrate implements this weighting via context-conditional gain on the policy logits and bottom-up precision on the sensory pathway."

**What the paper claims.** That the project's substrate is the first computational instantiation of *one-modulator-many-EFE-terms* in an interoceptive RL agent, with the behavioural readouts (information-seeking under high observation noise, risk-averse policy under high injury prior) being the active-inference signatures of this weighting. This requires the experiment-designer to add an information-action (the "look" action in v3-round P2) and is the cleanest discriminator between active inference and a pure Doya-NA-β reading at site C.

**Lineage anchors.** Friston, Rigoli et al. 2015 (active inference and epistemic value); Pezzulo, Rigoli & Friston 2015 (active inference for behaviour and embodied cognition); Parr & Friston 2017 (active construction of the visual world); Schwartenbeck et al. 2015 (dopaminergic midbrain encodes expected certainty about outcomes); Friston, Da Costa et al. 2021 (sophisticated inference).

**Why this satisfies the user's constraints.** "Possibility" framing: the paper proposes that the substrate *could be read* as active inference, with explicit caveats about the framework's normative status. "No one-to-one mapping": the EFE decomposition gives a single weighted score, and the modulator's effect is on the *weight* of that score — by construction, one signal affects all three EFE terms, with the relative effect depending on context.

### Framing B3 — Heteroscedastic-uncertainty FiLM (the project's already-built compound)

**Tagline.** "Row EE-6's FiLM-Ensemble + heteroscedastic-precision compound *is* the Bayesian-brain implementation — the ensemble approximates the posterior over neuromodulator effects, the heteroscedastic head is the predicted precision on predictions, and the FiLM read-outs are the action of that precision on the policy. This is the cleanest possible no-one-to-one framing."

**Theoretical commitment.** Modest, and *already-built* in the project's concept memo and v4 §3.0. The compound's lineage is established (FiLM ⊂ Hypernet ⊂ point-mass-BHN-limit; Claim 4 of [film_hypernet_bnn_lineage_and_critic_modulation.md §2](../../concepts/film_hypernet_bnn_lineage_and_critic_modulation.md)). The paper says: "the project's substrate is an approximate Bayesian Hypernetwork with a heteroscedastic likelihood; the ensemble members produce a finite-sample posterior over $(\gamma, \beta)$ — i.e., over neuromodulator effects on the main network — and the heteroscedastic head produces predicted *precision* on the predictions themselves. This is one signal — the modulator state — producing a *distribution* over multiple effects, satisfying the no-one-to-one constraint by construction (the ensemble disagreement is the locus of multi-target effect heterogeneity)."

**What the paper claims.** That the compound (Row EE-6 in [film_neuromod_integration.md §5.4](../../concepts/film_neuromod_integration.md)) is the cleanest available *computational instantiation* of "single modulator state produces a posterior over many downstream effects" — without committing to active inference or predictive coding as a normative framework. The Bayesian-brain reading enters via Kendall & Gal 2017's heteroscedastic loss being mathematically identical to the Friston 2010 free-energy precision-update equation ($\partial F/\partial s_i = \tfrac{1}{2}(\pi_i \varepsilon_i^2 - 1)$, $\pi_i^* = 1/\mathbb{E}[\varepsilon_i^2]$ — see [active_inference_hypervigilance.md §1.2](../../concepts/active_inference_hypervigilance.md)).

**Lineage anchors.** Kendall & Gal 2017 (heteroscedastic likelihood); Krueger et al. 2017 (Bayesian hypernetworks); Friston 2010 (precision update); Turkoglu et al. 2022 (FiLM-Ensemble); plus the project's own [film_hypernet_bnn_lineage_and_critic_modulation.md](../../concepts/film_hypernet_bnn_lineage_and_critic_modulation.md) Claims 1–5.

**Why this satisfies the user's constraints.** "Possibility" framing: the compound is *demonstrated to work* (the lineage is rigorous; the gradients are coherent per the §10 audit); the bridge it suggests is a possibility, not a closed proof. "No one-to-one mapping": the ensemble produces a posterior, so the modulator state corresponds to a *distribution over effects*, not a single effect — biology's reality of "one neuromodulator, distributed receptor cascades, many downstream consequences" maps onto "one modulator state, ensemble of $(\gamma, \beta)$ settings, distribution of policy effects".

### My recommendation between the three

**Frame B1 as the primary; cite B3 as the architectural realisation; defer B2 to the discussion.** The reasoning:

- B1 is the *minimum theoretical commitment that earns the impactful framing*. It says "the project's substrate is a one-signal-many-precision-effects machine; this is the natural abstraction for the user's two constraints; the paper demonstrates the possibility" — without forcing the paper to take a normative stance on active inference.
- B3 is the *concrete instantiation* B1 references. The paper's architecture section can lean on B3's lineage (the FiLM ⊂ Hypernet ⊂ BHN-limit chain) to show that the substrate is non-trivially Bayesian-brain-shaped at the implementation level.
- B2 is the *strongest possible read* and is the right thing for the discussion to gesture toward, but committing the paper to active inference as a normative framework requires the information-action experiment, which the user has flagged as not in the current testbed.

This three-tier structure also lets the paper survive cleanly under either reviewer pressure direction: a sceptical reviewer who wants "less Friston, more demonstration" gets the B3-shaped substrate result and the B1 framing-light read; a Bayesian-brain-enthusiast reviewer who wants "more Friston, less standard RL" gets the B2 discussion gesture.

---

## §C — Position on the impactful-vs-reasonable trade-off

The user has named the trade-off as the symposium's central question. The Bayesian-brain lens has a *uniquely flexible* position on this spectrum, because the precision-coding family supports a range of commitment levels — from "precision is a useful descriptive term" (B1, low commitment) to "active inference is the normative framework for the agent's behaviour" (B2, high commitment).

**The impactful pole.** Position the paper as the first **computational instantiation of Bayesian-brain precision-modulation in an interoceptive RL agent**. This is a *strong* claim — it commits the paper to active inference as the framework, requires the information-action experiment to discriminate from a pure Doya-NA-β reading, and stakes the paper's contribution at the intersection of two literatures (computational psychiatry and biologically-inspired RL) that have very few computational bridges. Headline finding: "we demonstrate that one neuromodulatory-like signal, learned end-to-end, produces channel-selective precision-weighting that satisfies four signatures of active-inference hypervigilance (sign, lag, selectivity, timescale) in an interoceptive agent." This is **publishable in *Neural Computation*, *Computational Psychiatry*, or a high-tier theory venue** if the experiments land — but it requires the project to commit early and lose flexibility if a signature falls.

**The reasonable pole.** Frame the project as a *FiLM-substrate demonstration of behavioural-effect-heterogeneity from a single signal* — compatible with multiple Bayesian-brain readings but not requiring formal commitment to any one. Headline finding: "we demonstrate that a single modulator state, fanned out via FiLM to three computational sites, produces dissociable behavioural effects (regime-change recovery, channel-selective hypervigilance, plasticity gating) that suggest the possibility of a unified precision-modulation abstraction for biological neuromodulation in interoceptive RL." This is **publishable in NeurIPS, ICLR, or a biology-RL crossover venue** at any point the substrate works, without staking the paper on which precision-coding reading is correct.

**My recommendation: the reasonable pole, with the impactful pole in the discussion.** Specifically:

1. **The paper's headline claim should be at the reasonable pole** — substrate-level demonstration that one signal produces multi-target heterogeneous effects, with FiLM as the operationalisation, with biological-neuromodulator labels presented as one of several compatible interpretive overlays rather than as the framework's skeleton. **Frame B1 + B3** as I described in §B.
2. **The paper's discussion should explicitly raise the impactful reading** as a research-programme suggestion — "this substrate is consistent with the active-inference reading of biological neuromodulation as precision-on-expected-free-energy weighting; future work could test this by adding an information-action and measuring whether the modulator tracks epistemic-vs-pragmatic gradient changes at regime boundaries." **Frame B2** as gesture, not claim.
3. **The paper's title should describe what the agent does, not which framework the authors prefer.** Per the active-inference concept memo's prior recommendation ([active_inference_hypervigilance.md §5](../../concepts/active_inference_hypervigilance.md)), neither "precision" nor "active inference" belongs in the title; they earn their keep in the body and discussion. A working title in the form "Channel-selective behavioural effects from a single neuromodulator-like signal in an interoceptive RL agent" satisfies both constraints.

**Why not the impactful pole.** The impactful claim depends on the information-action experiment that the project does not yet have. The user has signalled the project is "still in planning" — committing the paper to a framework whose discriminating test is not in the current testbed is a structural fragility I would advise against.

**Why not the absolute-reasonable pole.** A paper that *only* claims "FiLM works for interoceptive RL" without any theoretical framing leaves the cellular-gain ↔ behavioural-hyperparameter bridge on the table. The user's first constraint ("impactful for integrating neural-gain-like algorithms with hyperparameter modulation") explicitly asks for the bridge to be *suggested*. B1 supplies the framing-light suggestion without forcing the bridge to be closed.

This is **the position I would defend in the symposium**: B1 + B3 in the body, B2 in the discussion, no normative-framework commitment in the title.

---

## §D — The non-trivial Bayesian-brain impact claim that uniquely earns the impact half

The reasonable framing is publishable; the impactful framing is not yet defensible. **What is the smallest non-trivial impact claim the Bayesian-brain lens supplies that the FiLM-as-channels framing alone does not?**

**The claim.** *The project's substrate may be the first interoceptive-RL implementation of one-signal-many-precisions, which is a key prediction of the Bayesian-brain framework that has been difficult to test computationally.* In compact form: the precision-coding tradition has predicted for over two decades that a single biological neuromodulator can implement *level-specific precision-weighting* across a hierarchical generative model (Friston 2005, 2010; Adams, Shipp & Friston 2013) — but no published computational model has demonstrated this on an end-to-end-trained RL agent with multi-modal interoceptive observations.

**Why this claim is non-trivial.** Three reasons:

1. **The level-specific precision prediction is theoretical, not computational, in the literature.** Adams, Shipp & Friston 2013 derives the precision-message-passing scheme; Shine 2021 Box 3 names the precision-as-gain identity; Bastos et al. 2012 maps it to canonical microcircuits. None of these is a working RL agent with multi-modal interoception. The project's substrate would be the first.
2. **The "single signal, many effects" property is the load-bearing one for a Bayesian-brain reading.** Yu & Dayan 2005 and Pezzulo, Rigoli & Friston 2015 both invoke this property; both lack a computational substrate that demonstrates it. The closest published precedent is Costacurta et al. 2024 Fig. 3F (dissociation-by-ablation of latent dimensions in NM-RNN), but Costacurta's substrate is a *low-rank weight-scaling* hypernetwork on a neuroscience task, not an interoceptive RL agent with channel-selective sensory modulation.
3. **Interoception is where the precision-coding tradition has been most under-tested computationally.** Seth & Friston 2016 ("Active interoceptive inference and the emotional brain"); Pezzulo 2014 ("Why do you fear the bogeyman? An embodied predictive coding model of perceptual inference"); Pezzulo & Vlaev 2017 ("Anticipating, explaining, and modulating pain") — all propose interoceptive precision-modulation accounts of affective state and pain. The computational evidence base is sparse. The project's substrate is positioned to *contribute to it*.

**What the paper needs to show.** Two pieces of evidence, both within the v4 §5 prediction program:

- **(D-1) Dissociable per-site responses to dissociable triggers.** The v3-round P3 / v4 §5.6.1 per-site partial regression on (observation-noise, state-prediction-error, value-prediction-error). The level-specific-precision prediction is that the three sites' γ outputs load disproportionately on their level-matched regressors. **This is the operational signature of "one signal, many precision-effects".** If the matrix of regression coefficients across sites × triggers is approximately diagonal, the precision reading is supported; if all sites load equally on all triggers, the modulator is a one-channel signal with redundant heads (Vecoven-style) and the precision claim retreats.
- **(D-2) Channel-selective effects on threat-relevant vs threat-irrelevant input modalities.** The v4 §5.2 site-A modality-specific γ statistics. The active-inference concept memo's central derivation ([active_inference_hypervigilance.md §3.2](../../concepts/active_inference_hypervigilance.md)) shows that channel-selective gain is what distinguishes precision-modulation from volume-control: a scalar broadcast through a uniform receptor density cannot produce $\Delta\gamma_{\mathcal{T}} \ne \Delta\gamma_{\mathcal{N}}$, but a multi-output read-out matrix on top of $h_{\text{mod}}$ can. The substrate-level demonstration that **biology gets one signal and many effects via heterogeneous receptor density** — and that the project's read-out matrix is the computational analogue — is the cleanest possible no-one-to-one demonstration.

**The falsifier.** If both (D-1) and (D-2) fail — the per-site regression matrix is dense (all sites load on all triggers), and $\gamma_A$ rises uniformly across modalities post-injury — the precision-coding interpretation is **refuted**; the substrate is doing one-channel work with passenger heads, and the paper retreats to the bare "FiLM-substrate for behavioural-effect heterogeneity" reading without the precision-coding framing. **This is a publishable negative finding** — it tells the field that the substrate is not yet rich enough to instantiate level-specific precision-weighting and identifies the architectural extensions that would be required (e.g., the T/P split as a prerequisite, per v4 §5.5).

**The cost of the claim.** Modest. The two pieces of evidence are already in the v4 §5 prediction program; the framing change is in *how the results are presented*, not what experiments are run. The substantive risk is that (D-1) fails because the modulator is doing single-channel work — but if so, the *reasonable* framing (B1 + B3 without the impact claim) still publishes.

---

## §E — Three new ideas the symposium has not previously considered

Each of these is new — not in v1–v4, not in the v3-round Bayesian-brain sidecar, not in the active-inference concept memo. Each takes 1–2 paragraphs to land.

### E1 — Reread the R2 win as evidence for precision-weighted behavioural readout, not as continual-learning recovery

**The idea.** The project's empirical anchor — the modulator beat the unmodulated baseline by approximately 25× the seed-noise floor on the second return to a previously-seen task stage ([memory insight 20260513_0014](../../../../.claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md)) — is currently read as a hyperparameter-modulation readout in v4 (re-framed away from continual-learning recovery per v4 §1). The Bayesian-brain reading offers a *third* interpretation: **R2 win is the behavioural signature of a precision shift on the return-to-task prior**. When the agent returns to a previously-seen task stage, an unmodulated agent must re-infer that "this is the same task" from observations; a modulated agent's $h_{\text{mod}}$ can carry a *running estimate of context-precision* that is high on the second return (the modulator has accumulated evidence that context is stationary), driving the policy precision $\Pi_\pi$ at site C up faster than the baseline can recover by observation alone. **The R2 win, on this reading, is the substrate's first evidence that the modulator implements a context-precision empirical-Bayes prior**, in the Friston 2010 / Adams, Shipp & Friston 2013 hierarchical-Bayes sense.

**Why this matters for the paper.** Three reasons. (a) The reading is *consistent with* the v4 hyperparameter-modulation reading but goes one level deeper — it explains *why* a hyperparameter shift recovers faster, namely because the precision-on-context-prior was already elevated by the modulator's recurrent state. (b) The reading is *testable* on the existing data: the modulator's $h_{\text{mod}}$ trajectory during R1 should correlate with the policy's post-R2 entropy recovery rate — a single regression on already-logged quantities. (c) The reading is *non-trivial* — it positions the R2 win as a Bayesian-brain phenomenon (empirical-prior precision shift) rather than as a generic RL phenomenon (faster recovery from one bag of weights vs another).

### E2 — A specific active-inference test under expected free energy gradients at regime change

**The idea.** Under active inference, the modulator's response at a regime change should track *the gradient of expected free energy with respect to the modulator state*, not the value-prediction error magnitude $|\delta|$ alone. Specifically, decompose the regime-change modulator burst into two components: $(\partial G_{\text{pragmatic}} / \partial h_{\text{mod}})$ — how much the modulator state affects expected reward under the new prior preference $p(o\mid C)$ — and $(\partial G_{\text{epistemic}} / \partial h_{\text{mod}})$ — how much the modulator state affects expected information gain about hidden states. **Under the AI framing, the burst should be the sum of these two gradients**, with their *ratio* depending on whether the regime change is informationally surprising (epistemic-dominant) or affectively dispreferred (pragmatic-dominant). This makes a *quantitative* prediction that the FiLM-as-Doya reading does not: the regime-change burst is decomposable, with the decomposition depending on environmental properties.

**Operationalisation.** In the v4 testbed, construct two regime changes with matched $|\delta|$: (a) a *high-information-gain change* (a context switch where a previously-unobserved state dimension becomes observable — epistemic-dominant); (b) a *high-affect change* (a context switch where injury risk rises sharply with no new observable — pragmatic-dominant). The AI prediction is that the modulator burst at (a) is larger than at (b) when the agent is *uncertain* about hidden state, and smaller than at (b) when the agent is *confident* about hidden state. A pure Doya-NA-β reading predicts no such asymmetry (the burst should track $|\delta|$ alone). **This is a one-paragraph experiment that adds one new task variant and one new logged quantity (the value-head distribution variance before and after the change)**; it is the cleanest possible discriminator between active inference and Doya-NA at the regime-change burst. It is also *not on the critical path* — if Frame B1 + B3 is the headline, this experiment lives in the discussion as a "next step that would distinguish B2 from B1".

### E3 — Frame the "possibility bridge" as biology's one-signal-many-precisions, computationally demonstrated

**The idea.** The user has named two constraints (possibility-not-coverage; no-one-to-one mapping) that need a coherent positive framing in the paper. The Bayesian-brain lens supplies one: **biology gives one signal multiple effects via heterogeneous precision-weighting across hierarchical levels, and the project's substrate is one computational demonstration of this principle — not a theory of it, not a claim about biology, just a demonstration that the principle is computationally instantiable**.

**The framing's load-bearing structure.** Three claims that together earn the impact half of the trade-off:

1. **Biology's reality.** A single biological neuromodulator (ACh, NA, DA, 5-HT) does not act on one functional channel — it produces heterogeneous effects across cortical layers, sensory streams, motor pathways, and prefrontal-decision circuits, with the effect at each site depending on receptor density, cellular type, and current network state (Marder 2012 *Annu Rev Neurosci*; Shine 2021 §"Cellular mechanisms"). The one-to-one mapping the early Doya 2002 framework proposed is **explicitly an abstraction**, acknowledged as such by the field's own current synthesis (Mei et al. 2022 §4).
2. **The precision-coding framework gives the abstraction principled structure.** Friston 2010 + Adams, Shipp & Friston 2013 + Kanai et al. 2015 + Shine 2021 jointly argue that biology's one-signal-many-effects pattern is *not* messy biological complexity; it is the natural action of a single precision-control signal across levels of a hierarchical generative model. **The same precision can act on a sensory stream, a state-transition stream, or a policy stream, depending on which level's receptor-density / read-out matrix it encounters**.
3. **The project's substrate computationally instantiates this principle.** One $h_{\text{mod}}$, three FiLM read-outs (per-feature scale and shift matrices), three behavioural effects. The paper's contribution is to *demonstrate that the principle works* — that an end-to-end-learned modulator can implement one-signal-many-precisions in an interoceptive RL agent — not to claim that this is how the brain does it.

**Why this framing is the user's two constraints satisfied simultaneously.** "Possibility" framing: the paper demonstrates that the principle is computationally instantiable; it does not claim the bridge to biology is closed. "No one-to-one mapping": the framing *starts* from the no-one-to-one observation as a feature of biology and *builds* the substrate to honour it. The Bayesian-brain framing is therefore not just compatible with the user's constraints — **it is the natural articulation of the user's intuition**.

---

## Cross-references

- v4 direction memo: [docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md) — primary input; §3 unification claim; §3.0 lineage; §5 prediction program.
- Concept memo: [docs/project/concepts/film_neuromod_integration.md](../../concepts/film_neuromod_integration.md) — §3.0 lineage callout; §5.4 Row EE-6; §10 math-reviewer audit; §5.5 no-map boundary.
- Investigation memo: [docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md](../../concepts/film_hypernet_bnn_lineage_and_critic_modulation.md) — §2 Claims 1–5; §3 critic-modulation options; §3.5 Option A recommendation.
- Active inference concept memo: [docs/project/concepts/active_inference_hypervigilance.md](../../concepts/active_inference_hypervigilance.md) — §1.2 precision-update equation; §1.3 expected free energy; §1.4 site-mapping table; §3 the degenerate fixed point; §5 headline-term verdict.
- v3-round Bayesian-brain sidecar: [docs/project/references/neuromodulatory_algorithms/feedback/20260516_bayesian_brain_predictions_for_v3.md](../../references/neuromodulatory_algorithms/feedback/20260516_bayesian_brain_predictions_for_v3.md) — P1–P7 predictions; P3 level-specific precision; P4 d′-vs-criterion; P7 ACh-branch re-opening.
- Predictions synthesis: [docs/project/references/neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md](../../references/neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md) — §3.4 precision predictions; §3.5 unification predictions; §3.6 the missing bidirectional test.

## Hand-offs

- **`research-postdoc`** — owns the synthesis of professor positions. From the Bayesian-brain side, the load-bearing input is the §C recommendation (B1 + B3 body, B2 discussion) and the §D non-trivial impact claim. If other professors converge on a similar shape, the symposium's recommendation to the user becomes clean.
- **`pi`** — owns the focus-vs-explore call. The Bayesian-brain side's position is that the *reasonable* pole earns publishability without forcing the project to commit to active inference; the *impactful* pole is achievable but depends on the information-action experiment that is not yet in the testbed. This is a decision the user must own.
- **`experiment-designer`** — does NOT need new experiments for the §C recommendation. The §D claim rides on v4 §5.6.1 (per-site partial regression) and §5.2 (channel-selective γ statistics), which are already in the prediction program. The §E2 active-inference discriminator would require one new task variant and one new logged quantity; it is *optional* and deferred to discussion-level "next steps" unless the project commits to Frame B2.
- **`senior-developer`** — no code or config changes required for the §C recommendation. The §E1 R2-reread is a post-hoc regression on already-logged quantities.

---

*Contribution by `professor-bayesian-brain`, 2026-05-16.*
