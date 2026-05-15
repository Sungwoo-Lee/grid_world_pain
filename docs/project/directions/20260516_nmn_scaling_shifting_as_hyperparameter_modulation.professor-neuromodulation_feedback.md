---
title: "Feedback from professor-neuromodulation on 'NMN as scaling-and-shifting...' direction memo"
parent: docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation.md
reviewer: professor-neuromodulation
date: 2026-05-16
---

## Overall verdict — the unification needs work, not rejection

The big move the memo is making is to read the project's modulator network (a small side-circuit that watches the agent's state and outputs scale/shift signals into the main policy) as **simultaneously** (a) the *Doya 2002* story — each neuromodulator system tunes one knob of an RL algorithm — and (b) the *neural-gain* story — neuromodulators act on the slope and operating point of every target neuron's input-output curve. My verdict: **the unification is defensible at the level of "one signal moves both an algorithmic knob and a population-of-neurons' gain", but the memo as written overstates how *clean* the bridge is and silently elides three real gaps** — a scalar-vs-population mismatch, a multiplicative-vs-additive distinction the corpus is actually careful about, and a single-`mod_h` bottleneck that biology partitions across four separate ascending systems. The redirection itself ("continual-learning recovery is a downstream symptom of a hyperparameter shift, not the mechanism") is consistent with the strongest reading of *Lee et al. 2024*, *Mei et al. 2022*, and *Doya 2002*. But the proposed architecture is one signal doing four jobs, and that needs either the *Phase 0 T/P split* or an explicit caveat. Concrete corrections below.

### Q1 — Level mismatch (population vs scalar)

There is a real gap, and the memo papers over it. *Doya 2002* §2.3 maps each modulator to one scalar metaparameter (α, β, γ, δ). *Ferguson and Cardin 2020* (Box 1) and *Shine et al. 2021* (Fig. 1) both define neural gain as the slope `dQ/dI` of a *single neuron's* input-output curve — population-distributed, with cell-type-specific gain on E vs I cells, and target-projection-specific gain (Ferguson & Cardin §"Cellular mechanisms"). FiLM with $\gamma, \beta \in \mathbb{R}^d$ is closer to *Vecoven et al. 2020*'s $\sigma^{NMN}(x, z; w_s, w_b) = \sigma(z^T(xw_s + w_b))$ — a *scalar* neuromodulatory $z$ producing *per-neuron* $w_s$, $w_b$ via learned linear read-outs. That is the bridge: **`mod_h` is the scalar (Doya-side), and the per-channel FiLM read-out is the population dimension (gain-side).** This is biologically defensible (one nucleus, one scalar tone, fanning out to a population of targets whose receptor density determines per-cell effect), but the memo never says this. Recommendation: §3 should add one sentence — "`mod_h` is the Doya-side scalar; FiLM read-outs are the receptor-density-analog fan-out". Otherwise a reviewer reads "FiLM = Doya scalar" and rejects on the spot.

### Q2 — Tsuda 2021 fit

The memo cites *Tsuda et al. 2021* as authority for the additive-shift arm. **This citation is misapplied.** Tsuda's mechanism (their Fig. 1b, Methods, and Extended Data Appendix A) is **multiplicative scaling of synaptic weights** ($W \to f_{nm} W$), and Tsuda explicitly distinguishes their model from the "neural excitability" models of Stroud and *Vecoven et al. 2020* (Tsuda 2021 p. 2, line 50: "Our model of synaptic weight modulation shares some similarities to previous models, particularly to the neural excitability models [19, 20]. Both lead to increased flexibility and versatility, yet they operate through *independent mechanisms both biologically and computationally* (see Extended Data Appendix A)"). The "hypertube shift" in their Fig. 3 is a *consequence* in activity space of multiplying weights — not an additive shift on activations. So:

- Tsuda's mechanism is closer to *Costacurta et al. 2024* (low-rank weight scaling) and *Rodriguez-Garcia et al. 2026* (effective-weight $W = g \cdot w$) — the **multiplicative arm**, not the additive one.
- The FiLM-on-activations machine in the project is closer to Vecoven/Stroud — and Tsuda's own authors are careful to say these are not the same.
- The unification therefore cannot lean on Tsuda for the additive arm. The memo's prediction (b) ("ablating $\beta$ preserves Tsuda hypertube") is, on a strict reading of Tsuda, **backwards**: hypertube separation in Tsuda comes from the multiplicative arm.

Recommendation: rewrite §3 and prediction (b) — the FiLM-on-activations does need the *additive* arm for an operating-point shift, but the citation for that shift should be *Ferguson and Cardin 2020*'s additive/subtractive box (Box 1, panel c — rheobase shift via input current bias), not Tsuda.

### Q3 — Doya channel attribution

The memo's three injection sites (A perception, B memory, C decision-making) already map to three Doya channels in `project_plan.md` §3.1: A → ACh-like (precision), B → 5-HT-like patience / weak NE-tonic, C → 5-HT aversive + DA tonic. **Driving all three from a single `mod_h` recurrent state forces correlated movement on knobs that biology demonstrably dissociates.** *Doya 2002* §3 is explicit that the four systems are anatomically distinct (Fig. 1) with different projection patterns and different teaching signals (DA from VTA/SNc reward-prediction error; ACh from basal forebrain attention/uncertainty; NE from LC unexpected-uncertainty; 5-HT from dorsal raphe long-horizon). *Lee et al. 2024* implements this as **two separately-parameterised modulators** for ACh and NA, each with its own teaching rule. A single `mod_h` is closest to **NA on the unexpected-uncertainty reading** (because the project's empirical anchor — a burst on regime change — is the canonical NA signature, *Aston-Jones and Cohen 2005*, *Wainstein et al. 2025*), but C-site temperature modulation is also NA-flavoured under Doya 2002 §3.3. So the single-`mod_h` is plausibly NA-like at A and C, and **forced to overload as 5-HT-like at B**. This is the strongest case for Q6.

### Q4 — Compatibility with the continual-learning corpus

The redirection ("continual learning is a symptom, not the mechanism") is **stronger than the memo claims**, and is actively supported by *Lee et al. 2024* §1 and *Mei et al. 2022* (their explicit framing is "neuromodulatory processes may fine-tune *hyperparameters* … to dynamically adapt learning strategies of DNNs" — Figure 1B), and by *Durstewitz et al. 2025*'s list of continual-learning mechanisms (their §"Four major categories" — replay, regularization, modularity, partial resets — *and* gain neuromodulation in *Rodriguez-Garcia et al. 2026* sits explicitly in the optimizer-dynamics category, not the architectural one). The corpus does **not** force a "neuromodulation = continual-learning aid" framing. So the redirection sits well with this literature, with one caveat: the strongest continual-learning result in the corpus (*Rodriguez-Garcia et al. 2026* on stability gap) is itself reframed as gain-as-hyperparameter, not gain-as-continual-learning-machine. Recommendation: §4 should cite Lee 2024 §1 and Mei 2022 Fig. 1B for the redirect; the corpus supports it.

### Q5 — Differentiation from Rodriguez-Garcia 2026

This is the load-bearing concern, and **the differentiation is real but the memo does not yet make it**. *Rodriguez-Garcia et al. 2026* §2.1 derives the unification cleanly: $W_{ij}(t) = g_i(t) w_{ij}(t)$, then decomposes phasic + tonic, and proves that gain-modulated SGD is mathematically a two-timescale optimizer with curvature reparameterisation $\lambda \to \lambda/g^2$. Their gain is **multiplicative-only**, acts on **weights** (not activations), and is driven by an explicit **uncertainty surprise signal $H(y) \propto |\partial_g L|$**. The project's FiLM is **multiplicative + additive**, acts on **activations** (not weights), and is driven by a **learned recurrent state $h_{\text{mod}}$ with no explicit uncertainty teaching signal**. So the project's claim differentiates on three axes:

1. Additive arm — operating-point shift, not just slope.
2. Activation-level, not weight-level — closer to Vecoven 2020, which Tsuda and Rodriguez-Garcia both distinguish themselves from.
3. State-dependent (recurrent), not surprise-locked — the modulator can sustain a tonic shift between regime changes, which Rodriguez-Garcia's $g(t) = \gamma g(t-1) + (1-\gamma)g_0 + \eta H$ cannot.

The unification then makes a sharper prediction than Rodriguez-Garcia: the project's modulator should ride **both** a phasic burst at the regime change *and* a sustained tonic shift between regimes — which is exactly what the *Phase 0 T/P split* is for. Recommendation: §3 of the memo should add this three-axis contrast, and prediction (a) should be split into a phasic-burst test and a tonic-shift test.

### Q6 — Single-`mod_h` bottleneck

**Yes, the unification requires the T/P split** (`project_plan.md` §3.2). Three independent reasons:

1. *Doya 2002* §3 partitions across four anatomically distinct systems; the project's three injection sites already need at least two of those (NA-like at A/C, 5-HT-like at B).
2. *Rodriguez-Garcia et al. 2026* §2.1 derives the fast/slow decomposition $W = w_{\text{slow}} + w_{\text{fast}}$ from a phasic + tonic gain split — a single time-constant `mod_h` cannot produce this without an architectural T/P split.
3. The project's *§3.2 hysteresis* claim (T enters a high-tonic near-absorbing state) is the chronic-pain analog the project plan already commits to; a single recurrent state with one time constant cannot implement it.

Without the T/P split, the unification claim collapses to "the modulator does NA-like phasic", which is *Wainstein 2025* + *Rodriguez-Garcia 2026*, and the project's contribution shrinks to "we did it on an interoceptive RL task". Recommendation: §6 of the memo should escalate Q6 from "open question" to **prerequisite** — the unification frame must be conditional on the T/P split landing.

### Q7 — Falsification

The cleanest staked test is **prediction (d) — modulator burst tracks $|\delta|$ (or value-head-residual proxy) at regime change, not just a switch indicator**. Here is why prediction (b) is weaker:

- Prediction (b) (ablate $\gamma$ vs $\beta$) is a *mechanism-isolation* test that confirms the FiLM machinery works, but cannot distinguish "this is gain control" from "this is learned context-conditioning". *Vecoven et al. 2020* and *Tsuda et al. 2021* would both pass it; *Doya 2002* makes no commitment.
- Prediction (d) directly stakes the **Doya-channel claim**: if the modulator burst is regressed on the TD-error (or critic-residual) magnitude, a significant correlation supports the NA-as-inverse-temperature / DA-as-TD-error reading. A *null* correlation — modulator burst keyed only to task-id / regime label — refutes the Doya-channel reading and demotes the modulator to a context-detector, leaving only the (already-published, *Vecoven 2020*) FiLM-as-context-conditioning claim.
- Stronger still: combine (d) with the T/P split. P should track $|\delta|$ phasically; T should track cumulative cost / value-baseline. A *clean dissociation* is the unification's strongest empirical signature.

Stake the headline claim on prediction (d). Keep (b) as a mechanism check, not a falsifier.

### Three minor flags (no need to fix before sister-reviewer sign-off)

- **Missing citations in the corpus**: the memo cites Xing 2022, Kudithipudi 2022, and Wang 2024, but these PDFs are not in `docs/project/references/neuromodulatory_algorithms/sources/`. If they are load-bearing for §2, get them into the corpus before the next round.
- **Memo says "ACh ↔ signal-to-noise / precision"** (§2). Doya 2002 §3.4 is "ACh ↔ learning rate α". Yu and Dayan 2005 is the source for ACh ↔ precision. Cite Yu & Dayan 2005 for the precision reading; Doya alone won't support it.
- **`mod_h` is unitless and unconstrained**, but Rodriguez-Garcia 2026 explicitly bounds $g(t) \geq g_0 \geq 1$ (gain is multiplicative, $\geq 1$ baseline). If the project wants to differentiate as "gain-like", FiLM's $\gamma$ should probably be constrained to be positive (e.g., softplus). Hand-off: this is a `senior-developer` / `experiment-designer` item for the Phase 0 build.

### Hand-offs

- **`senior-developer`** — re-verify §3.2 T/P split is on the Phase 0 critical path; constrain FiLM $\gamma > 0$ if "gain" is to be the headline word.
- **`experiment-designer`** — re-derive the four-probe program around prediction (d) as falsifier and (a) split into phasic/tonic tests.
- **`research-postdoc`** — synthesise this feedback with `professor-rl-bayesian-dl`'s when both arrive; the joint revision is the next round.

---
*Feedback by `professor-neuromodulation`, 2026-05-16.*
