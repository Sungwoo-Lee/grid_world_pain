---
title: "NMN as scaling-and-shifting implementation of RL-hyperparameter modulation"
status: draft-with-feedback
audience: user, professor-neuromodulation, professor-rl-bayesian-dl, pi, experiment-designer
last_updated: 2026-05-16
supersedes: docs/project/directions/nmn_continual_lifelong_probe.md
related:
  - docs/project/directions/nmn_meta_context_conditioning_v2.md
  - docs/project/ideas/20260515_continual_learning_probe_program.md
  - docs/experiments/summaries/20260513_0321_nmn_comparison_study.md
empirical_anchor: .claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md
---

# NMN as scaling-and-shifting implementation of RL-hyperparameter modulation

## 1. What this direction proposes

This memo redirects the project's neuromodulation work away from "the **NMN** (a small side-network that watches state and outputs scaling/shifting signals into the main policy) helps **continual learning** (training where the environment changes across stages)" toward a stronger claim: **the NMN's algorithm — multiplicatively scale and additively shift the policy's internal activations — is simultaneously the cellular-level account of how biological neuromodulators work (gain control on target neurons) and the behaviour-level account of how reinforcement-learning hyperparameters (learning rate, exploration temperature, future-reward discount, signal-to-noise weighting) get modulated state-by-state**.

A **FiLM** layer (feature-wise linear modulation: $h \mapsto \gamma(c) \odot h + \beta(c)$, the simplest context-dependent scale-and-shift) is the bridge: the same operation is gain control at the activation level and a hyperparameter knob at the behaviour level. Our empirical anchor — the modulator beat the unmodulated baseline by ~25× the seed-noise floor on returns to a previously-seen stage ([memory insight 20260513_0014](../../../.claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md)) — is reread as "NMN implements a state-dependent hyperparameter shift, and continual-learning recovery is one behavioural readout of that shift", not "NMN helps continual learning".

## 2. The two lineages

| Lineage | Unit of analysis | What modulators do | Anchor citations |
|---|---|---|---|
| **Doya — RL hyperparameter modulation** (behaviour level) | The agent | DA ↔ learning rate / TD error; ACh ↔ signal-to-noise / precision; NA ↔ inverse-temperature / exploration; 5-HT ↔ discount on future reward. A scalar output re-tunes an algorithmic knob. | Doya 2002; Xing 2022, Ben-Iwhiwhu 2022, Lee 2024, Wang 2024. |
| **Neural-gain — scaling and shifting on activations** (cellular / network level) | The neuron / population | Multiplicative scaling of input-output curves, additive shifts of operating point, changed response variability. | Ferguson and Cardin 2020; Shine 2021; Vecoven 2020 (FiLM in DNNs), Tsuda 2021 (gain shifts RNN activity hypertubes), Wainstein 2025, Rodriguez-Garcia 2026 (NA-gain attenuates stability gap), Costacurta 2024. |

Doya-lineage papers rarely commit to a cellular implementation; neural-gain papers rarely commit to a behaviour-level hyperparameter mapping. They have been argued in parallel.

## 3. The unification claim

A single FiLM layer applied to a recurrent policy's activations is **both stories at once**. At the activation level the multiplicative arm scales the downstream layer's input-output curve; the additive arm shifts its operating point — gain modulation in the Ferguson-Cardin / Tsuda / Costacurta sense. At the behaviour level the same scalars re-parameterise effective RL hyperparameters: scaling the actor's logits is an inverse-temperature change (NA); scaling encoder pre-fusion activations is a precision / signal-to-noise change (ACh); shifting the GRU update gate is a memory-stability / effective-discount change (5-HT, with caveats). The project's modulator already injects at all three sites (Injection A on the encoder, B on the memory GRU, C on policy temperature) — **the architecture we already have is, on this reading, simultaneously a multi-site gain controller and a multi-channel hyperparameter modulator**. This is a re-interpretation, not a new architecture proposal.

## 4. What this redirects the project away from

The R2 continual-learning win (the FiLM agent surviving +107 / +132 steps on the two returns to a previously-seen active-predator stage, ~25× the seed-noise floor) was the anchor for the prior framing, which read it as "the NMN helps continual learning". That framing is too narrow: the win is one behavioural readout of a state-dependent hyperparameter shift the modulator implements at every regime change. The four-probe continual-learning program ([tracker](../ideas/20260515_continual_learning_probe_program.md)) inherits this reframe — peri-boundary parameter freeze, expressivity-matched control, long dormancy still measure useful quantities, but get re-derived as **probes of the unification claim**, not of "does NMN help continual learning". Re-derivation is the next round's work; this memo does not re-scope them.

## 5. What the new direction predicts

- **(a) State-dependent exploration shift.** At a regime change the modulator's burst coincides with a near-instantaneous policy-entropy spike — effective temperature rises for tens of episodes, then relaxes. NA / inverse-temperature reading, from logged actor entropy + `mod_h`.
- **(b) Multiplicative vs. additive arm dissociation.** Ablating $\gamma$ only (freeze at 1, keep $\beta$) preserves the additive-shift / hypertube readout (Tsuda 2021) but eliminates gain-rescaling (Rodriguez-Garcia 2026); ablating $\beta$ only does the converse. The unified frame predicts *both* contribute.
- **(c) Doya-channel attribution from per-site clamps.** Injection A clamp degrades precision behaviours (noisy obs); Injection C clamp degrades exploration-required behaviours (regime changes); Injection B clamp degrades tasks needing protected memory across switches.
- **(d) Modulator burst correlates with $|\delta|$, not just task-id.** Activity should track absolute TD error (or value-head-residual proxy) at regime change, not just a switch indicator — cheapest test of "Doya channel" vs. "label detector".
- **(e) No win where the testbed cannot dissociate the two stories.** Steady-state heterogeneous-noise environments need no hyperparameter movement, so the gain machinery has nothing to do — the v8 null is expected, and converts retroactively into evidence for the unified frame.

## 6. Open questions for the professors

1. **Level mismatch.** Doya's mapping is one scalar per modulatory system; cellular gain modulation is population-distributed. Is our multi-dimensional FiLM ($\gamma, \beta \in \mathbb{R}^d$) genuinely the same machine as a 1-knob hyperparameter modulator, or is there a population-vs-scalar gap the unification papers over?
2. **Tsuda 2021 fit.** Does the hypertube framing fit our combined scale-and-shift, or is it specifically an *additive-shift* story that does not need the multiplicative arm? If the latter, the unification must be argued at the joint level (gain + hypertube), not from a single mechanism.
3. **Doya channel attribution.** `mod_h` is one recurrent state feeding three injection sites. Which Doya channel does it primarily correspond to — NA (Injection C), ACh (Injection A), or a mixture? Is a single-`mod_h` modulator forced to entangle channels in a way biology does not?
4. **Compatibility with the "NMN aids continual learning" corpus.** Kudithipudi 2022, Durstewitz 2025, Lee 2024 frame neuromodulation as a continual-learning aid. Does our redirection ("continual learning is a symptom, not the mechanism") sit with that framing or warn us off it?
5. **Differentiation from Rodriguez-Garcia 2026.** Their NE-gain-attenuates-stability-gap proposal is structurally close to ours. Is our claim differentiable, or is the unification essentially what their gain-on-effective-weights argument already implies?
6. **Single-`mod_h` bottleneck.** If the unification is real, each Doya channel should be independently controllable — but a single scalar `mod_h` forces correlated movement. Does the unification *require* the Phase 0 T/P split (`project_plan.md` §3.2)?
7. **Falsification.** What single experiment would force us to accept that scaling-and-shifting is gain control OR hyperparameter modulation but not both? Predictions (b) and (d) are candidates; which would you stake the claim on?

## 7. References and hand-offs

- **Empirical anchor**: R2 continual sister-pair finding, [memory insight 20260513_0014](../../../.claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md). Plain-English re-summary in the [study report](../../experiments/summaries/20260513_0321_nmn_comparison_study.md).
- **Reference corpus** (full set to be checked by the professors): `docs/project/references/neuromodulatory_algorithms/sources/`. Load-bearing for §2: Doya 2002, Xing et al. 2022, Ben-Iwhiwhu et al. 2022, Lee et al. 2024, Wang et al. 2024 (Doya lineage); Ferguson and Cardin 2020, Shine et al. 2021, Vecoven et al. 2020, Tsuda et al. 2021, Wainstein et al. 2025, Rodriguez-Garcia et al. 2026, Costacurta et al. 2024 (neural-gain lineage). Additional corpus items (Durstewitz et al. 2025, AlKilany and Goodman 2025, Mei et al. 2022, Osman et al. 2024, Tambaş et al. 2025, Driscoll et al. 2022, Wang et al. 2025) are not directly cited above but may be relevant to the open questions.
- **Superseded**: [`nmn_continual_lifelong_probe.md`](nmn_continual_lifelong_probe.md). The architecture-vitality argument and the five-stage continual schedule are absorbed; the *framing* of the win as "continual-learning resistance" is what this memo walks back.
- **Adjacent**: [`nmn_meta_context_conditioning_v2.md`](nmn_meta_context_conditioning_v2.md). The 2×3 meta-context probe is compatible — under the unification frame it reads as "does the modulator factorise across two Doya-channel-distinguishable axes?", a sharper question than the v2 memo's P3 factorisation claim.
- **Re-scoped on landing**: [`20260515_continual_learning_probe_program.md`](../ideas/20260515_continual_learning_probe_program.md) — the four probes inherit the redirection; `experiment-designer` re-derives them after this direction is accepted.
- **Next hand-offs**: professor-neuromodulation and professor-rl-bayesian-dl review the unification claim and fill §8 below; on acceptance, `experiment-designer` re-scopes the probe program around predictions §5 (a)–(e); on disagreement, `research-postdoc` synthesises and surfaces the divergence.

## 8. Professor feedback

> Reviews from `professor-neuromodulation` and `professor-rl-bayesian-dl`, each having read the postdoc's draft and the project's reference corpus at `docs/project/references/neuromodulatory_algorithms/`. **Both professors independently flagged the Tsuda 2021 mis-citation in §3 and prediction (b); `professor-rl-bayesian-dl` additionally flagged a Doya/Yu-Dayan branch confusion in the §2 ACh row.** These are the load-bearing fixes to weigh before the postdoc revises.

### Feedback from professor-neuromodulation

#### Overall verdict — the unification needs work, not rejection

The big move the memo is making is to read the project's modulator network (a small side-circuit that watches the agent's state and outputs scale/shift signals into the main policy) as **simultaneously** (a) the *Doya 2002* story — each neuromodulator system tunes one knob of an RL algorithm — and (b) the *neural-gain* story — neuromodulators act on the slope and operating point of every target neuron's input-output curve. My verdict: **the unification is defensible at the level of "one signal moves both an algorithmic knob and a population-of-neurons' gain", but the memo as written overstates how *clean* the bridge is and silently elides three real gaps** — a scalar-vs-population mismatch, a multiplicative-vs-additive distinction the corpus is actually careful about, and a single-`mod_h` bottleneck that biology partitions across four separate ascending systems. The redirection itself ("continual-learning recovery is a downstream symptom of a hyperparameter shift, not the mechanism") is consistent with the strongest reading of *Lee et al. 2024*, *Mei et al. 2022*, and *Doya 2002*. But the proposed architecture is one signal doing four jobs, and that needs either the *Phase 0 T/P split* or an explicit caveat. Concrete corrections below.

##### Q1 — Level mismatch (population vs scalar)

There is a real gap, and the memo papers over it. *Doya 2002* §2.3 maps each modulator to one scalar metaparameter (α, β, γ, δ). *Ferguson and Cardin 2020* (Box 1) and *Shine et al. 2021* (Fig. 1) both define neural gain as the slope `dQ/dI` of a *single neuron's* input-output curve — population-distributed, with cell-type-specific gain on E vs I cells, and target-projection-specific gain (Ferguson & Cardin §"Cellular mechanisms"). FiLM with $\gamma, \beta \in \mathbb{R}^d$ is closer to *Vecoven et al. 2020*'s $\sigma^{NMN}(x, z; w_s, w_b) = \sigma(z^T(xw_s + w_b))$ — a *scalar* neuromodulatory $z$ producing *per-neuron* $w_s$, $w_b$ via learned linear read-outs. That is the bridge: **`mod_h` is the scalar (Doya-side), and the per-channel FiLM read-out is the population dimension (gain-side).** This is biologically defensible (one nucleus, one scalar tone, fanning out to a population of targets whose receptor density determines per-cell effect), but the memo never says this. Recommendation: §3 should add one sentence — "`mod_h` is the Doya-side scalar; FiLM read-outs are the receptor-density-analog fan-out". Otherwise a reviewer reads "FiLM = Doya scalar" and rejects on the spot.

##### Q2 — Tsuda 2021 fit

The memo cites *Tsuda et al. 2021* as authority for the additive-shift arm. **This citation is misapplied.** Tsuda's mechanism (their Fig. 1b, Methods, and Extended Data Appendix A) is **multiplicative scaling of synaptic weights** ($W \to f_{nm} W$), and Tsuda explicitly distinguishes their model from the "neural excitability" models of Stroud and *Vecoven et al. 2020* (Tsuda 2021 p. 2, line 50: "Our model of synaptic weight modulation shares some similarities to previous models, particularly to the neural excitability models [19, 20]. Both lead to increased flexibility and versatility, yet they operate through *independent mechanisms both biologically and computationally* (see Extended Data Appendix A)"). The "hypertube shift" in their Fig. 3 is a *consequence* in activity space of multiplying weights — not an additive shift on activations. So:

- Tsuda's mechanism is closer to *Costacurta et al. 2024* (low-rank weight scaling) and *Rodriguez-Garcia et al. 2026* (effective-weight $W = g \cdot w$) — the **multiplicative arm**, not the additive one.
- The FiLM-on-activations machine in the project is closer to Vecoven/Stroud — and Tsuda's own authors are careful to say these are not the same.
- The unification therefore cannot lean on Tsuda for the additive arm. The memo's prediction (b) ("ablating $\beta$ preserves Tsuda hypertube") is, on a strict reading of Tsuda, **backwards**: hypertube separation in Tsuda comes from the multiplicative arm.

Recommendation: rewrite §3 and prediction (b) — the FiLM-on-activations does need the *additive* arm for an operating-point shift, but the citation for that shift should be *Ferguson and Cardin 2020*'s additive/subtractive box (Box 1, panel c — rheobase shift via input current bias), not Tsuda.

##### Q3 — Doya channel attribution

The memo's three injection sites (A perception, B memory, C decision-making) already map to three Doya channels in `project_plan.md` §3.1: A → ACh-like (precision), B → 5-HT-like patience / weak NE-tonic, C → 5-HT aversive + DA tonic. **Driving all three from a single `mod_h` recurrent state forces correlated movement on knobs that biology demonstrably dissociates.** *Doya 2002* §3 is explicit that the four systems are anatomically distinct (Fig. 1) with different projection patterns and different teaching signals (DA from VTA/SNc reward-prediction error; ACh from basal forebrain attention/uncertainty; NE from LC unexpected-uncertainty; 5-HT from dorsal raphe long-horizon). *Lee et al. 2024* implements this as **two separately-parameterised modulators** for ACh and NA, each with its own teaching rule. A single `mod_h` is closest to **NA on the unexpected-uncertainty reading** (because the project's empirical anchor — a burst on regime change — is the canonical NA signature, *Aston-Jones and Cohen 2005*, *Wainstein et al. 2025*), but C-site temperature modulation is also NA-flavoured under Doya 2002 §3.3. So the single-`mod_h` is plausibly NA-like at A and C, and **forced to overload as 5-HT-like at B**. This is the strongest case for Q6.

##### Q4 — Compatibility with the continual-learning corpus

The redirection ("continual learning is a symptom, not the mechanism") is **stronger than the memo claims**, and is actively supported by *Lee et al. 2024* §1 and *Mei et al. 2022* (their explicit framing is "neuromodulatory processes may fine-tune *hyperparameters* … to dynamically adapt learning strategies of DNNs" — Figure 1B), and by *Durstewitz et al. 2025*'s list of continual-learning mechanisms (their §"Four major categories" — replay, regularization, modularity, partial resets — *and* gain neuromodulation in *Rodriguez-Garcia et al. 2026* sits explicitly in the optimizer-dynamics category, not the architectural one). The corpus does **not** force a "neuromodulation = continual-learning aid" framing. So the redirection sits well with this literature, with one caveat: the strongest continual-learning result in the corpus (*Rodriguez-Garcia et al. 2026* on stability gap) is itself reframed as gain-as-hyperparameter, not gain-as-continual-learning-machine. Recommendation: §4 should cite Lee 2024 §1 and Mei 2022 Fig. 1B for the redirect; the corpus supports it.

##### Q5 — Differentiation from Rodriguez-Garcia 2026

This is the load-bearing concern, and **the differentiation is real but the memo does not yet make it**. *Rodriguez-Garcia et al. 2026* §2.1 derives the unification cleanly: $W_{ij}(t) = g_i(t) w_{ij}(t)$, then decomposes phasic + tonic, and proves that gain-modulated SGD is mathematically a two-timescale optimizer with curvature reparameterisation $\lambda \to \lambda/g^2$. Their gain is **multiplicative-only**, acts on **weights** (not activations), and is driven by an explicit **uncertainty surprise signal $H(y) \propto |\partial_g L|$**. The project's FiLM is **multiplicative + additive**, acts on **activations** (not weights), and is driven by a **learned recurrent state $h_{\text{mod}}$ with no explicit uncertainty teaching signal**. So the project's claim differentiates on three axes:

1. Additive arm — operating-point shift, not just slope.
2. Activation-level, not weight-level — closer to Vecoven 2020, which Tsuda and Rodriguez-Garcia both distinguish themselves from.
3. State-dependent (recurrent), not surprise-locked — the modulator can sustain a tonic shift between regime changes, which Rodriguez-Garcia's $g(t) = \gamma g(t-1) + (1-\gamma)g_0 + \eta H$ cannot.

The unification then makes a sharper prediction than Rodriguez-Garcia: the project's modulator should ride **both** a phasic burst at the regime change *and* a sustained tonic shift between regimes — which is exactly what the *Phase 0 T/P split* is for. Recommendation: §3 of the memo should add this three-axis contrast, and prediction (a) should be split into a phasic-burst test and a tonic-shift test.

##### Q6 — Single-`mod_h` bottleneck

**Yes, the unification requires the T/P split** (`project_plan.md` §3.2). Three independent reasons:

1. *Doya 2002* §3 partitions across four anatomically distinct systems; the project's three injection sites already need at least two of those (NA-like at A/C, 5-HT-like at B).
2. *Rodriguez-Garcia et al. 2026* §2.1 derives the fast/slow decomposition $W = w_{\text{slow}} + w_{\text{fast}}$ from a phasic + tonic gain split — a single time-constant `mod_h` cannot produce this without an architectural T/P split.
3. The project's *§3.2 hysteresis* claim (T enters a high-tonic near-absorbing state) is the chronic-pain analog the project plan already commits to; a single recurrent state with one time constant cannot implement it.

Without the T/P split, the unification claim collapses to "the modulator does NA-like phasic", which is *Wainstein 2025* + *Rodriguez-Garcia 2026*, and the project's contribution shrinks to "we did it on an interoceptive RL task". Recommendation: §6 of the memo should escalate Q6 from "open question" to **prerequisite** — the unification frame must be conditional on the T/P split landing.

##### Q7 — Falsification

The cleanest staked test is **prediction (d) — modulator burst tracks $|\delta|$ (or value-head-residual proxy) at regime change, not just a switch indicator**. Here is why prediction (b) is weaker:

- Prediction (b) (ablate $\gamma$ vs $\beta$) is a *mechanism-isolation* test that confirms the FiLM machinery works, but cannot distinguish "this is gain control" from "this is learned context-conditioning". *Vecoven et al. 2020* and *Tsuda et al. 2021* would both pass it; *Doya 2002* makes no commitment.
- Prediction (d) directly stakes the **Doya-channel claim**: if the modulator burst is regressed on the TD-error (or critic-residual) magnitude, a significant correlation supports the NA-as-inverse-temperature / DA-as-TD-error reading. A *null* correlation — modulator burst keyed only to task-id / regime label — refutes the Doya-channel reading and demotes the modulator to a context-detector, leaving only the (already-published, *Vecoven 2020*) FiLM-as-context-conditioning claim.
- Stronger still: combine (d) with the T/P split. P should track $|\delta|$ phasically; T should track cumulative cost / value-baseline. A *clean dissociation* is the unification's strongest empirical signature.

Stake the headline claim on prediction (d). Keep (b) as a mechanism check, not a falsifier.

##### Three minor flags (no need to fix before sister-reviewer sign-off)

- **Missing citations in the corpus**: the memo cites Xing 2022, Kudithipudi 2022, and Wang 2024, but these PDFs are not in `docs/project/references/neuromodulatory_algorithms/sources/`. If they are load-bearing for §2, get them into the corpus before the next round. *(Top-level note: in fact all three PDFs are present in the corpus; this flag is a reviewer mis-read of a partial directory listing.)*
- **Memo says "ACh ↔ signal-to-noise / precision"** (§2). Doya 2002 §3.4 is "ACh ↔ learning rate α". Yu and Dayan 2005 is the source for ACh ↔ precision. Cite Yu & Dayan 2005 for the precision reading; Doya alone won't support it.
- **`mod_h` is unitless and unconstrained**, but Rodriguez-Garcia 2026 explicitly bounds $g(t) \geq g_0 \geq 1$ (gain is multiplicative, $\geq 1$ baseline). If the project wants to differentiate as "gain-like", FiLM's $\gamma$ should probably be constrained to be positive (e.g., softplus). Hand-off: this is a `senior-developer` / `experiment-designer` item for the Phase 0 build.

##### Hand-offs

- **`senior-developer`** — re-verify §3.2 T/P split is on the Phase 0 critical path; constrain FiLM $\gamma > 0$ if "gain" is to be the headline word.
- **`experiment-designer`** — re-derive the four-probe program around prediction (d) as falsifier and (a) split into phasic/tonic tests.
- **`research-postdoc`** — synthesise this feedback with `professor-rl-bayesian-dl`'s when both arrive; the joint revision is the next round.

*Feedback by `professor-neuromodulation`, 2026-05-16.*

### Feedback from professor-rl-bayesian-dl

#### Headline verdict

The unification claim — "FiLM-style multiplicative scale + additive shift is *simultaneously* gain control on activations and a state-dependent RL-hyperparameter knob" — is **partially correct but currently overstated**. It is correct that the *substrate* (a $\gamma, \beta$ pair injected at three sites) supports both readings. It is overstated that any single FiLM layer with a shared scalar `mod_h` instantiates *all four* Doya channels (ACh, NA, DA, 5-HT) coherently. Two specific weaknesses need surgery before the memo can carry a falsifiable claim. First, the table in §2 misstates the Doya 2002 mapping for ACh (Doya assigns ACh to learning-rate $\alpha$, not signal-to-noise / precision — the precision reading is **Yu & Dayan 2005**, a different paper, and the corpus doc set actually does not contain Xing et al. 2022 as cited). Second, the project is structurally closest to **Rodriguez-Garcia et al. 2026**, which differs from FiLM in a load-bearing way the memo does not flag: their gain acts on the **gradient update** (an optimiser-level mechanism), whereas FiLM acts on the **forward activation** (a function-approximator-level mechanism). These are not the same machine. Below I work through the seven questions in order, with the most consequential being Q3 (channel attribution, where I think the project's `mod_h` is currently a **temperature-modulator with two passenger outputs**) and Q5 (Rodriguez-Garcia is a precedent, not a near-twin).

#### Question-by-question response

##### Q1 — Level mismatch (FiLM $\gamma, \beta \in \mathbb{R}^d$ vs scalar hyperparameter)

There is a real gap, but it is not fatal — it is the gap between **per-unit gain control** (the Ferguson & Cardin 2020 picture, where every cortical neuron has its own slope) and **per-channel hyperparameter signalling** (the Doya 2002 picture, where one scalar tunes one knob). The unification has to be made at the **read-out level**: a $d$-dimensional $\gamma$ at injection C (policy logits) reduces to a 1-D inverse-temperature **only after the readout matrix** averages or projects. For a single-output softmax head whose logits are $\gamma \odot W h + \beta$, the effective inverse temperature is $\|\gamma\|$-scaled along the policy axis, not $\gamma$ itself. So the per-feature $\gamma$ is *over-parameterised* relative to the Doya knob — which is fine if the optimiser collapses the extra degrees of freedom, and problematic if it does not. **Hypernetwork comparison.** The over-parameterisation is the same critique that distinguishes Vecoven et al. 2020's $z \in \mathbb{R}^k$ from a true 1-knob modulator: Vecoven's Fig. 7 explicitly shows that many of their $z$-dimensions go un-recruited on simple benchmarks; the recruitment patterns are diagnostic of how many effective scalars the network actually uses. **Recommendation:** at evaluation time, compute the effective rank of $\gamma$ across regime changes — if it collapses to rank 1 at the policy site, the per-feature FiLM is doing 1-knob work and the Doya claim holds; if it stays high-rank, FiLM is doing something Doya doesn't describe.

##### Q2 — Tsuda 2021 fit (hypertube as additive-only?)

The memo gets this backwards. **Tsuda 2021 is explicitly a multiplicative-weight-modulation story** ("uniform multiplicative factor acting on synaptic weights", §Results, Fig. 1b), not an additive-shift story. The hypertubes in PCA space *look* like shifts, but the mechanism producing them is multiplicative on the recurrence matrix $W$ — they appear as additive in activity-space because $\Delta x \propto (g-1) W x$ for the modulation factor $g$. So the joint $\gamma + \beta$ FiLM is *narrower* than Tsuda — Tsuda modulates $W$ itself (weights), FiLM modulates $h$ (activations). **Costacurta et al. 2024 §3** sits between them: their NM-RNN scales the **rank-1 factors of a low-rank recurrence matrix**, which is multiplicative on weights via a structured factorisation, and they show this is equivalent to an LSTM forget gate (their Prop. 1). The hypertube reading is therefore best argued as: "the multiplicative arm $\gamma$ on the GRU pre-activation is a structured-flexibility analogue of Tsuda's weight scaling, with $\beta$ contributing the activation-bias hypertube origin shift". The memo should not predict (b) — "ablating $\gamma$ preserves the additive-shift / hypertube readout" — because Tsuda's hypertubes are themselves the *consequence* of multiplicative weight scaling, not the additive arm.

##### Q3 — Doya channel attribution for `mod_h`

This is the most surgically actionable question. A single recurrent `mod_h` feeding three sites is **almost certainly behaving as a Noradrenaline/inverse-temperature signal with two passenger heads**. Reasoning: (i) Injection C (policy temperature) is the only site where the modulation has a **direct readable RL-hyperparameter consequence** — the actor's $\beta = f(\text{mod}_h)$ literally rescales the logits, exactly Doya 2002 §3.3. (ii) Injection A (encoder pre-fusion) is a precision-weighting candidate (Yu & Dayan 2005's ACh-as-expected-uncertainty, **not** Doya's ACh-as-learning-rate), but only if the projection from `mod_h` to the encoder is rank ≥ 2 across channels — otherwise it collapses to a global volume knob that the policy can absorb back into the temperature. (iii) Injection B (GRU update gate) is the cleanest 5-HT/discount analogue *if* it actually modulates the update gate (which shortens or lengthens effective memory and so changes effective $\gamma$), but a single scalar shifting the GRU update gate is **not** a discount-factor knob in any clean sense — the GRU update gate is per-unit and acts on the recurrent state, not on the value-function horizon. **Empirical test:** correlate `mod_h` activity with (a) actor-entropy spikes at regime change, (b) value-prediction-error magnitude, (c) effective-horizon measured by autocorrelation of `mod_h_GRU_state`. If (a) dominates, `mod_h` is a Noradrenaline analogue; if all three are equally strong, the **single-`mod_h` design is entangling channels biology keeps separate** — which is exactly the Phase 0 T/P split's motivation (project_plan.md §3.2).

##### Q4 — Compatibility with the "NMN aids continual learning" corpus

The redirection does **not** warn the project off Kudithipudi 2022 / Durstewitz 2025 / Lee 2024. Lee et al. 2024 is in fact the **paper closest to the project's redirected claim** — its §3.1 explicitly proposes that ACh and NA modulate RL **hyperparameters** (learning rate and exploration), and its Figure 1 gives an Anaconda diagram of "hypothesised connection between neuromodulator and RL hyperparameter" that is almost the project's diagram. Where Lee differs: it is a **non-stationary multi-armed bandit** with the modulator outputting a learning-rate adjustment via an estimated unexpected-uncertainty signal; the project differs by (i) operating in a sequential RL setting (PPO actor-critic, not bandit), (ii) using FiLM at three sites rather than just $\alpha, \beta$ scalars, and (iii) the modulator being end-to-end learned rather than a hand-engineered functional form (Lee's "component IV"). The redirection should **explicitly cite Lee 2024 as the closest precedent** and frame the project's contribution as "Lee's framework extended to a multi-site FiLM substrate, with the hyperparameter mapping emerging from optimisation rather than from a hand-coded functional form". The "continual learning is a symptom" framing is then compatible — it is the empirical readout that Lee 2024 doesn't fully exploit.

##### Q5 — Differentiation from Rodriguez-Garcia 2026

This is the **most important architectural correction**. Rodriguez-Garcia 2026 is **not** what the memo currently implies. Their gain $g(t)$ multiplies the **gradient update** (their Eq. 4 + Algorithm 1: $g(t+1) = \gamma g(t) + (1-\gamma) g_0 + \eta H$, where $H$ is the network entropy, *then* the SGD step uses $g(t) \cdot \nabla_W L$ effectively because $\partial L / \partial w = g \cdot \partial L / \partial W$). This is an **optimiser-level** mechanism — it makes the learning rate state-dependent. The project's FiLM, by contrast, is a **function-approximator-level** mechanism — it modifies the forward pass, leaving the optimiser untouched (PPO with Adam, learning rate constant). These produce *different* signatures: Rodriguez-Garcia flattens the Hessian via reparameterisation ($\lambda \to \lambda / g^2$, their §2.1); FiLM does not. So the unification claim is **not** what Rodriguez-Garcia's argument already implies — they are arguing that gain modulation is equivalent to a **fast-slow weight decomposition** (their Eq. 2), which is a *learning-rate* claim, while the project's claim is about *forward-time* hyperparameter modulation. The two could be **complementary**: a future version of the project could add a Rodriguez-Garcia-style optimiser-level gain on the modulator weights themselves while keeping the forward-time FiLM. Differentiation language for the memo: "Rodriguez-Garcia 2026 modulates the **gradient** at task boundaries; we modulate the **forward computation** at every step. Both invoke noradrenergic gain, but at orthogonal points in the training-and-inference pipeline."

##### Q6 — Single-`mod_h` bottleneck and the T/P split

The single-`mod_h`-to-three-sites design is **provably non-identifiable** at the channel level: any rotation of the three site-specific output heads that preserves their projection onto policy-relevant subspaces is observationally equivalent. The Phase 0 **T/P (tonic/phasic) split** referenced in project_plan.md (§3.2) — if it gives the modulator two recurrent states with separable timescales — is necessary for the unification claim to be testable. Without it, the claim "NMN implements multi-channel hyperparameter modulation" is degenerate with the claim "NMN implements one-channel modulation with redundant heads". The connection to Costacurta 2024 is direct: their NM-RNN ablation (their Fig. 3F) shows that ablating individual neuromodulatory dimensions has dissociable effects only *because* the network learned to use the dimensions differentially — the project should expect the analogous result only after the T/P split forces the modulator to maintain at least two timescales. **Hand-off:** `senior-developer` plan the T/P split as Phase 0 prerequisite; `experiment-designer` author per-injection-site ablations as the channel-attribution probe (predictions (c) in §5).

##### Q7 — Falsification

The cleanest single experiment is **prediction (b) sharpened**, not (d). The reason: prediction (d) ("modulator burst correlates with $|\delta|$") is **co-implied by both stories** — gain control and hyperparameter modulation both predict the modulator should care about prediction error, so (d) cannot dissociate them. Prediction (b) — multiplicative arm vs. additive arm dissociation — can, if reframed as I recommended in Q2. The sharpened version: **(b′) Freeze $\gamma=1$ at all sites; the network can still encode hypertube-shifts via $\beta$ (Tsuda-style activity-space displacement), but **loses the inverse-temperature / exploration arm** at Injection C** — because rescaling logits is a multiplicative-only operation. If the agent's regime-change recovery degrades and entropy-spike behaviour vanishes under $\gamma$-freeze but not under $\beta$-freeze, **the multiplicative arm is doing Doya-NA work**, and the unification holds *at site C*. If both freezes are equally damaging, the FiLM arms have entangled and the unification holds only in aggregate. If $\beta$-freeze is more damaging than $\gamma$-freeze, the project is doing **hypertube-shift dynamics in Tsuda's sense**, and the Doya/RL-hyperparameter framing should be walked back — gain control is the right story, hyperparameter modulation is not. This single ablation experiment is small (4 conditions × seeds at one stage of the existing testbed) and directly stake-able.

#### Load-bearing factual error flag (for postdoc before professor review lands)

The §2 table maps ACh to "signal-to-noise / precision". **Doya 2002 §3.4 maps ACh to learning-rate $\alpha$**, not precision. The signal-to-noise reading is from Yu & Dayan 2005 ("Uncertainty, neuromodulation, and attention", *Neuron*), which is a different paper and not in the project's reference corpus. This matters because the project's empirical predictions (notably (c) "Injection A clamp degrades precision behaviours") rest on which mapping is in force. **Recommendation to postdoc:** either (i) replace "signal-to-noise / precision" with "learning-rate" and re-derive prediction (c), or (ii) keep the precision reading and add Yu & Dayan 2005 as a co-anchor alongside Doya 2002, explicitly noting that ACh-as-precision is the "Yu-Dayan branch" and ACh-as-learning-rate is the "Doya branch" — the project is choosing the former. Lee et al. 2024 §3 (in-corpus) is a model for how to cite both branches cleanly. Also: the §7 reference list cites "Xing et al. 2022" but that paper is not present in `sources/`; the in-corpus near-substitutes are Ben-Iwhiwhu et al. 2022 (context meta-RL via neuromodulation) and Wang et al. 2024 (Neuromodulated Meta-Learning). *(Top-level note: Xing 2022 is in fact present in the corpus.)*

#### Hand-offs

- **`senior-developer`**: plan the T/P split (Q6) as a Phase 0 prerequisite if not already scheduled; the unification claim is non-identifiable without at least two modulator timescales.
- **`experiment-designer`**: author the $\gamma$-freeze vs. $\beta$-freeze ablation (Q7, prediction b′) at Injection C as the primary unification-vs-gain-control test. Per-site ablations (Q3) as the secondary channel-attribution probe.
- **`research-postdoc`**: fix the Doya/Yu-Dayan branch confusion in §2 (Q1 + factual flag) before the direction lands. Cite Lee et al. 2024 as the closest precedent (Q4) and Rodriguez-Garcia 2026 as a complementary, not overlapping, mechanism (Q5).

*Feedback by `professor-rl-bayesian-dl`, 2026-05-16.*

---

## 9. Convergence note from the two reviewers

Both professors are aligned on **load-bearing fixes** the postdoc needs to make in v2; they disagree on **which prediction to stake the headline claim on**. Surface for user feedback:

**Convergent fixes (both reviewers):**

1. **Tsuda 2021 is mis-cited.** Tsuda's mechanism is multiplicative-on-weights, not additive-on-activations. Prediction (b) as currently written is backwards. Citation for the additive arm should be Ferguson & Cardin 2020 (Box 1c rheobase shift) instead.
2. **Single-`mod_h` bottleneck is non-identifiable** without the Phase 0 T/P split. Q6 should be escalated from "open question" to "prerequisite". Senior-developer to re-verify T/P split is on the Phase 0 critical path.
3. **Differentiation from Rodriguez-Garcia 2026 is real but not yet in the memo.** They modulate the *gradient* / optimiser; the project modulates the *forward activation* / function approximator. Memo should add this contrast in §3.
4. **Lee et al. 2024 is the closest precedent**, not the corpus's other Doya-RL papers. §4 should cite Lee 2024 §1 + Fig. 1 explicitly as the framework the project extends.
5. **`mod_h` is likely behaving as Noradrenaline / inverse-temperature with two passenger heads** under the current single-recurrent-state design. The T/P split is what makes the multi-channel claim testable.

**Divergence — which prediction is the falsifier:**

- `professor-neuromodulation` stakes the headline on **prediction (d) — modulator burst tracks $|\delta|$ at regime change**, because it directly tests the Doya-channel reading. (b) is a mechanism check, not a falsifier.
- `professor-rl-bayesian-dl` stakes the headline on **prediction (b′) — $\gamma$-freeze vs. $\beta$-freeze at Injection C**, because (d) is co-implied by both gain-control and hyperparameter-modulation stories so cannot dissociate them.

Their disagreement is itself diagnostic: it tells us the falsification design needs to **combine** the two — a $\gamma$/$\beta$ ablation that ALSO checks whether the surviving arm's activity tracks $|\delta|$. The user should weigh whether the v2 memo commits to one falsifier or to the combined two-stage test.

**Minor flag (top-level note):** both professors reported corpus PDFs missing (Xing 2022, Kudithipudi 2022, Wang 2024). These PDFs are in fact present in `docs/project/references/neuromodulatory_algorithms/sources/`. The mis-flag was a reviewer-side directory-listing read error and has been noted inline. No action needed.
