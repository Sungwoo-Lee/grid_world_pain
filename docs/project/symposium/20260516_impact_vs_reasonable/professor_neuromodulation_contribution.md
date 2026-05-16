---
title: "Symposium contribution — biological-substrate position on impactful-vs-reasonable framing"
session: 2026-05-16_impact_vs_reasonable
contributor: professor-neuromodulation
date: 2026-05-16
inputs:
  - docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md
  - docs/project/concepts/film_neuromod_integration.md
  - docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md
  - docs/project/references/neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md
  - docs/project/references/neuromodulatory_algorithms/reviews/ferguson_cardin_2020_gain_modulation.md
prior_contributions:
  - docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation.md (v1 §8)
  - docs/project/references/neuromodulatory_algorithms/feedback/20260516_neuromod_predictions_for_v3.md (v3 round)
---

# Symposium contribution — biological-substrate position on impactful-vs-reasonable framing

## 1. Plain-English entry point

The user has asked the group to decide where on the *impactful-vs-reasonable* spectrum the paper should sit. The user's two design constraints are sharp: (i) the paper should suggest the *possibility* of bridging neural gain and behavioural-hyperparameter modulation, not claim the full bridge; (ii) the mapping must not be one-to-one, because biology's neuromodulators are demonstrably multifunctional.

The position I bring from the biological-substrate side is that **v4 still inherits the one-to-one channel commitment more than it acknowledges**. v4 anchors on "two cleanly carried channels (ACh-as-α at site A; NA-as-β at site C) plus one re-named third (plasticity gating at site B); 5-HT/γ out-of-substrate". That is a Doya 2002 four-channel reading with two slots filled, one renamed, one declared empty. The user's constraint cuts at a *different* axis than v4's narrowing — it says the channels themselves should not be expected to be one-to-one with biological systems. Real ACh does not "implement α"; real NA does not "implement β"; each neuromodulator nucleus produces multiple downstream effects depending on cortical layer, receptor type, and projection target.

My recommendation: the paper should be framed around **target-specific gain on a single substrate-level modulator**, not channel-specific hyperparameter assignment. The single `mod_h` reading three injection sites through three learned read-out matrices is *biologically more faithful than Doya's clean four-channel mapping*. The paper should claim the *possibility* of this bridge as a single demonstration of a broader biological principle (one upstream signal, many downstream effects via target-specific receptor density), with v4's two-channel reading kept as one *interpretive lens* (alongside an active-inference precision lens), not as the load-bearing structural claim.

---

## 2. §A — One-to-many biological evidence: the brain's neuromodulators are not single-function

The user's second constraint — *"our brain's neuromodulation is not affect to single functionality"* — has direct biological backing. Doya 2002's four-channel mapping is a *normative simplification*, not a description of how the brain actually deploys neuromodulators. Per system:

**Noradrenaline / locus coeruleus.** Aston-Jones & Cohen 2005 adaptive-gain theory: one neuron-class (LC) → many downstream effects depending on firing mode and target — arousal regulation, sensory signal-to-noise, attentional gating, exploration-exploitation arbitration (Doya-β), memory consolidation, descending pain modulation (LC → spinal dorsal horn). All from one nucleus. Wainstein 2025 (in corpus) shows one LC-NA gain signal driving multiple behavioural switches. A one-to-one "NA ↔ inverse-temperature β" cannot survive this list — temperature is *one* of NA's effects, not its function. Sara 2009 (canonical, out of corpus) reviews the multi-functionality.

**Acetylcholine / basal forebrain.** Ferguson & Cardin 2020 *Neuromodulatory control* at the receptor level: *"Layer-specific: ACh suppresses excitatory neurons in L4 but enhances them in L2/3 and L5."* Same projection, opposite effects depending on layer. ACh's downstream functions: learning-rate (Doya 2002 §3.4), sensory precision / expected-uncertainty (Yu & Dayan 2005), attentional gain (Sarter & Lustig), memory consolidation (Hasselmo), divisive-normalisation attention (Ferguson & Cardin Box 1d). One signal, many effects via receptor-density fan-out. The corpus's contested ACh branch (α vs precision; predictions synthesis §1) is *exactly* this multifunctionality — Doya and Yu-Dayan are not in conflict; they describe two of ACh's many effects.

**Dopamine / VTA-SNc.** Schultz / Montague / Dayan's RPE is textbook; Doya 2002 §3.1 itself acknowledges DA's role in vigor (Niv), motor gating, motivation, exploration. Friston's active-inference reinterpretation reads DA's phasic burst as *precision on policy beliefs* — a fifth effect.

**Serotonin / dorsal raphe.** Doya 2002 §3.2 maps 5-HT to γ; Daw, Kakade & Dayan 2002 to aversive opponent control; Dayan & Huys to reflective behavioural inhibition; Cohen et al. to dorsal-raphe patience. Lee 2024 §6 declines to instantiate 5-HT/γ because "no convergent normative theory exists" — the right reading is not "the field hasn't decided" but *5-HT implements many functions and the field hasn't decided which one Doya's framework should privilege*. The corpus-internal concession that one-to-one breaks at 5-HT first.

**Conclusion.** The brain's "one signal, many effects depending on target/context" is an *architectural feature*, not an entanglement bug. Doya 2002 is a normative starting point, not a faithful description.

**The reframing v4 still needs.** The project's substrate — single `mod_h`, three FiLM injection sites with separately-learned read-out matrices — is *the most biologically faithful design choice in the corpus on this point*. v4 §3 frames `mod_h` as the Doya-side scalar with "FiLM read-out matrix as the receptor-density-analog fan-out" — but then *spends* that credit by attempting a clean A=ACh, C=NA channel assignment. Under the genuine one-to-many reading, the three read-outs are *three targets in the receptor-density fan-out of one ascending modulator*, each target producing multiple effects: encoder gain → learning-rate-like + precision-like + divisive-normalisation-like simultaneously; policy-logit gain → temperature + action-prior + attentional-bias simultaneously; GRU-gate gain → plasticity + persistence + prior-precision simultaneously.

My v1 §8 Q3 critique — *"single `mod_h` forces correlated movement on knobs biology dissociates"* — needs revision in light of the user's framing. **Biology does not dissociate them the way Doya's framework suggests.** The substrate is *more* biologically faithful than the framework v4 attempts to inherit. The critique stands only against the strong claim "the substrate carries four cleanly dissociable Doya channels"; what v4 has not walked back is the framing that "the substrate carries channels at all" rather than "the substrate carries one signal whose effects are target-specific and multi-functional".

---

## 3. §B — Alternative biological framings for the "possibility bridge"

Three concrete framings the paper could adopt, all of which avoid one-to-one channel commitment while preserving the structural claim that gain-modulated forward-pass operators produce behavioural-hyperparameter-like effects.

### B.1 — Target-specific gain on a single ascending signal

*Framing.* "We demonstrate the *possibility* that a single neuromodulator-like ascending signal can produce multiple behavioural-hyperparameter-like effects via target-specific gain (FiLM read-out matrices as the in-silico analogue of receptor-density fan-out). The three injection sites are not three channels of a four-knob agent; they are three targets receiving the same upstream signal through different read-out matrices, and each target produces a *family* of behavioural effects."

*Biological grounding.* Ferguson & Cardin 2020 *Neuromodulatory control*: every neuromodulator listed (ACh, NA, 5-HT, DA) is described in terms of *which receptor subtypes act on which cell classes in which layers*. Shine 2021 (in corpus) formalises gain as $g = dQ/dI$ and traces it to systems-level signatures (low-D dynamics; energy-landscape flattening) — one cellular operation, many systems-level readouts. Wainstein 2025 (in corpus) anchors this empirically.

*What changes vs. v4.* v4 §3 says "site A = α channel; site C = β channel". Under B.1: "site A is *one target* of the modulator; what the modulator does at site A is mostly characterisable in α / precision / divisive-normalisation terms; alternative readings remain admissible". Same architecture, weaker channel commitment, more biologically defensible.

### B.2 — Multi-functional gain as a single demonstrated instance

*Framing.* "Neural gain modulation in the brain is multi-functional: one nucleus produces multiple downstream effects depending on receptor density. Our FiLM-on-PPO substrate is an *in-silico instance* of this principle — a single learned modulator producing behavioural effects that map onto multiple RL hyperparameters simultaneously. We do not claim full coverage; we claim a single demonstration that the bridge is constructable end-to-end."

*Biological grounding.* Same as B.1 plus the LC-NA multifunctionality literature (Aston-Jones & Cohen 2005; Sara 2009, canonical-out-of-corpus). The bridge claim is *demonstrated*, not *proved*.

*What changes vs. v4.* v4 §1 claim is *"the NMN's algorithm is simultaneously the cellular-level account of gain control and the behaviour-level account of how two of Doya's four channels get modulated"*. Under B.2: *"the same algorithm — gain modulation on activations — can simultaneously play multiple algorithmic roles, paralleling biological multifunctionality. The behavioural readouts at our three sites are interpretable through multiple algorithmic lenses (Doya hyperparameters; precision levels; Aston-Jones tonic-phasic); we do not commit to a single lens being correct."* The plurality of *interpretive* lenses is part of the claim.

### B.3 — Receptor-density-as-readout: the FiLM read-out matrix as biological structure

*Framing.* "The FiLM read-out matrices that project `mod_h` to per-feature $(\gamma, \beta)$ at each site are the in-silico analogue of receptor-density patterns. In the brain, a single neuromodulator (e.g., LC-NA) projects to many regions with different receptor densities ($\alpha_1$, $\alpha_2$, $\beta_1$, $\beta_2$ adrenergic), producing different effects on different cell classes and layers. In our architecture, the read-out matrices play the same structural role: one upstream signal, multiple target-specific effects."

*Biological grounding.* Ferguson & Cardin 2020 Box 1 + *Neuromodulatory control* anchored at receptor level (for NA: *"β-adrenergic receptors ↑ excitability; α-adrenergic receptors ↓ synaptic excitation"*). Receptor-density-as-readout is implicit in Shine 2021's *cellular-to-dynamics* framing.

*What changes vs. v4.* v4 §3 has a short *"scalar-to-population bridge"* paragraph that already names the receptor-density analogy — but it sits as a stylistic aside; the load-bearing structural claim is still Doya-channel-assignment. Under B.3, that paragraph is *promoted* to the load-bearing structural claim, with channel-assignment as the interpretive layer on top. Smallest edit to v4 that achieves the user's "no one-to-one mapping" constraint.

### Which framing the paper should adopt

B.3 is the smallest edit; B.1 the cleanest re-anchoring; B.2 the most honest "demonstration" framing constraint (i) asks for. I recommend **B.3 plus B.2 in combination**: structural claim is receptor-density-as-readout (B.3); rhetorical framing is single-demonstration of multifunctional gain (B.2); v4's two-channel Doya reading is preserved as *one of several interpretive lenses* in §5 rather than as the load-bearing commitment in §3. §5 already does this implicitly (each per-site prediction carries both a Doya-channel reading and a Bayesian-brain precision reading); the §3 unification claim needs to inherit the same plurality.

---

## 4. §C — Impactful-vs-reasonable position from the biological-substrate side

The two extreme positions the user set up:

- **Impactful (high-ambition).** "FiLM is THE substrate for biological neural gain — we prove a one-to-many mapping that subsumes Doya's framework."
- **Reasonable (low-ambition).** "FiLM is a demonstration that gain-modulated forward-pass operators can produce multiple behavioural-hyperparameter-like effects on a single substrate — one instance of a broader biological principle."

From the biological-substrate side I recommend the **reasonable end of the spectrum, but on a *biologically richer claim*** than v4's two-cleanly-carried-channels framing. Concretely: the paper claims (i) the *possibility* of bridging cellular-level gain and behavioural-hyperparameter modulation in a single learned substrate; (ii) that the bridge is *multi-functional* (one signal → multiple effects through target-specific read-outs); (iii) that the specific behavioural effects achievable are *broader* than Doya 2002's clean four-knob list — they cover Doya's hyperparameters in part, but also cover Bayesian-brain precision levels, Aston-Jones tonic/phasic modes, and target-class-specific gain in ways the four-knob framework alone does not anticipate.

**Why "reasonable but biologically richer" is the right position.**

*Reasonable on the unification claim.* v4's "two cleanly carried channels" already moved from v3's four-channel to a narrower scope; constraint (i) pushes one more step — to "possibility" rather than "carried channels". The right load-bearing claim is *demonstration that the bridge exists*, not *enumeration of which channels the substrate carries*. The empirical anchor (the R2 win in memory insight 20260513_0014) is a sufficient demonstration; v4 §5 then *characterises* the demonstration rather than *verifying a four-channel mapping*. Two framings of the same experiments.

*Biologically richer on the cellular-to-behavioural mapping.* v4 inherits Doya's four-knob limitation. Biology has more than four downstream readouts per neuromodulator. The paper can claim *more breadth* — multifunctionality is paradoxically more impactful than four-channel mapping — by *committing less* to the channel-assignment. Richer: "this substrate can carry effects spanning multiple algorithmic lenses simultaneously"; narrower: "this substrate carries two of Doya's four channels".

*Audience.* The cleanest biological reviewers (Friston-active-inference school, Daw, Niv) recognise the one-to-many framing immediately and push back on Doya-channel-assignment as a 23-year-old normative simplification the field has moved past. *Neural Computation* / *PLOS Comp Bio* / *eLife computational neuroscience* / CCN / COSYNE favour the multifunctional reading. Doya-channel commitment is more defensible at *NeurIPS* / biologically-inspired-RL workshops where Doya 2002 has canonical status — but is also more rebuttable there by biologically-minded reviewers. The trade-off favours one-to-many for the broader-audience venues.

**Specific recommendation: B.3 + B.2 framing, v4 experimental program preserved.** The experiments in v4 §5 are the right experiments. The framing should be re-cast as "characterising the multifunctionality of the substrate" rather than "verifying the two-channel Doya mapping". This is a §3 / §1 rewrite, not a §5 rewrite. The empirical critical path is unchanged.

---

## 5. §D — Two-three NEW ideas the symposium should consider

These are ideas I have not raised in v1 §8 or in the v3 sidecar. I keep them substrate-side; the other professors will propose complementary ideas from their lenses.

### D.1 — *"Modulator multifunctionality index"* as a new empirical readout

*Idea.* Define a single scalar — multifunctionality index $M$ — that counts how many distinct algorithmic-level behavioural effects `mod_h` produces simultaneously across the three sites. For each of $k$ candidate readings (Doya-α, Doya-β, ACh-precision, NA-tonic, NA-phasic, plasticity-gating, divisive-normalisation, action-prior-shift, exploration-bonus, etc.) compute the partial correlation between `mod_h` and the reading-specific empirical signature, controlling for the other $k-1$. $M$ = number of partial correlations above a pre-registered threshold.

*Why new.* v4 §5 is organised around per-site predictions, each with one Doya reading + one precision reading. $M$ *counts* readings without committing to a four-channel framework. $M=1$ = single-channel-with-passengers; $M=2$ = v4's two cleanly carried channels; $M\geq 5$ = the one-to-many regime constraint (ii) calls for.

*What it does.* Gives the paper a single headline number operationalising "multifunctionality". $M \geq 4$ on three sites driven by one `mod_h` is the strongest biological-plausibility claim *without* committing to which four functions are the four. Failure mode $M=1$ is still publishable ("even a substrate that *could* be multifunctional doesn't spontaneously become so under PPO optimisation").

### D.2 — *Target-specificity ratio* as the architectural payoff of multi-site injection

*Idea.* Compute, on the $3\times 3$ deficit matrix `eval/per_site_freeze_per_subtest_drop` (already in v4 §5.4), the diagonal-dominance ratio $T = (\text{diagonal sum})/(\text{off-diagonal sum})$, values $> 1$ meaning each site is preferentially responsible for its predicted subtest.

*Why new.* v4 §5.4 has the deficit matrix as a prediction; the new piece is *interpreting $T$ as the receptor-density-fan-out signature*, not the channel-dissociation signature. Under one-to-many, there are no channels to dissociate; what dissociates is the *target-specific effect of one signal*. Same measurement, biologically richer interpretation. The architectural claim becomes "we built a learned receptor-density-fan-out", not "we built a multi-channel modulator".

### D.3 — *Tonic/phasic dissociation as a multifunctionality test, not a Doya-mapping test*

*Idea.* v4 §5.5 frames the T/P split test as one-effect-per-timescale (phasic ↔ $|\delta|$; tonic ↔ cumulative cost). Under one-to-many, $\tau_T/\tau_P \geq 10$ separates *two timescales of modulator action*, and *each timescale produces multiple downstream effects*. Phasic should track $|\delta|$ AND drive policy-temperature spikes AND drive encoder-precision spikes AND drive plasticity-gating burst-mode. Tonic should track cumulative cost AND prior-precision elevation AND fear-avoidance entropy-reduction AND long-term plasticity shift.

*Why new.* v4 §5.5.2 stops at one effect per timescale (inherited from Aston-Jones tonic-phasic). Under one-to-many each timescale produces a *basket* of effects — a stronger test of multifunctionality at the timescale level. Aston-Jones & Cohen 2005 in fact describes tonic and phasic modes as producing multiple downstream effects each, not one each. The empirical test: phasic state's partial correlations with $\{|\delta|, \text{entropy spike at C}, \text{precision spike at A}, \tau_h \text{ burst at B}\}$ should *all* be positive simultaneously.

---

## 6. Cross-references with the other three professors

*Convergence.*

- **`professor-bayesian-brain`** will push the *active-inference precision-level reading* of the three sites (sensory $\Pi_s$ at A, state-transition $\Pi_z$ at B, policy $\Pi_\pi$ at C). Fully compatible with B.3 — precision-level is one of multiple interpretive lenses B.3 admits. We may diverge on whether precision should be the *headline* claim (their framing) or one of several lenses (mine). My D.1 multifunctionality index counts precision-level effects alongside Doya effects without privileging either.

- **`professor-rl-bayesian-dl`** will push the *FiLM ⊂ Hypernet ⊂ point-mass-BHN-limit* lineage. Load-bearing for the architectural side; less informative about the biological-substrate side. The biological reading of EE-6 (FiLM-Ensemble + Kendall-Gal) is *uncertainty-weighted multi-target gain*, which fits cleanly into B.3. From the publication side, architectural lineage is a *NeurIPS-shaped* contribution; biological multifunctionality is a *Neural Computation / eLife-shaped* contribution. The paper may need to pick which leads.

- **`professor-pain-modeling`** will push *construct-validity gate-keeping* (§5.7's four dissociations) and the *interoceptive-neuromodulation-as-substrate-for-pain-like-behaviour* framing. Fully aligned — pain construct validity is orthogonal to multifunctionality, and §A here strengthens the pain side (brain's chronic-pain machinery uses NA, ACh, opioidergic, 5-HT systems jointly, each contributing multiple effects — Eippert / Wager / Wiech descending opioidergic modulation is the cleanest example).

*Divergence.*

- vs. `professor-rl-bayesian-dl`: whether the EE-6 BHN-approximation framing should be load-bearing. Their architectural case is strong; from biology, BHN framing is *post-hoc rationalisation* of an architectural choice motivated by target-specific gain on a multifunctional modulator (B.3). My position: multifunctionality framing leads, BHN lineage supports.

- vs. `professor-bayesian-brain`: whether to keep Doya 2002 at all. Active-inference readers see Doya 2002 as the older anchor the field has partly superseded. From the substrate side I am *neutral* on dropping Doya — what I object to is one-to-one mapping. If `professor-bayesian-brain` wants to drop Doya, I would support it provided multifunctionality (D.3 multi-effect-per-timescale) is preserved.

---

## 7. Flag: is the v4 framing itself problematic from a biology standpoint?

Yes, partially. v4 has narrowed cleanly from v3 (math-reviewer audit on site B; FiLM-lineage §3.0; §5.9 honesty on γ_Bellman) but it still leads with a two-channel Doya mapping in §3 — and that two-channel mapping is itself a *one-to-one commitment* on the channels it *does* commit to. v4 says: *"Site A (encoder pre-fusion) — α channel"* and *"Site C (policy logits) — β channel"*. Under constraint (ii), *neither* should be a one-channel statement. Site A is plausibly *one target* producing a basket of effects (learning-rate-like, precision-like, divisive-normalisation-like, expected-uncertainty-like); site C is *one target* producing another basket (temperature-like, action-prior-like, attentional-bias-like, policy-precision-like, exploration-bonus-like). v4 picks one effect per site and labels it as *the* channel.

This is the *same one-to-one error* v4 just called out in Doya's framework, applied at the per-site level instead of per-channel. v4 walked back the Doya four-channel mapping but did not walk back the same error one level down.

**Recommendation to the postdoc who will synthesise the symposium**: v4's §3 unification claim should be re-anchored on B.3 (receptor-density-as-readout) + B.2 (single demonstration of multifunctional gain). The two-channel Doya reading from current §3 should move to a *§3.5 — Interpretive lenses* subsection listing three or four lenses (Doya hyperparameters; Bayesian-brain precision levels; Aston-Jones tonic-phasic modes; receptor-density target-specific gain) without privileging any one as load-bearing.

v4's experimental program (§5) is unchanged under this reframing — per-site clamps, γ/β arm dissociation, cross-site correlation structure, Hessian probe, construct-validity gate — all of these *characterise the multifunctionality* of the substrate just as well as they *verify the two-channel mapping*. Rhetorical, not experimental, but the rhetorical move constraints (i) and (ii) directly demand.

---

*Contribution by `professor-neuromodulation`, 2026-05-16.*
