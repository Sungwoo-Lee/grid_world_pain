# Direction — Continual / lifelong probe of the NMN as architecture-vitality check

**Author**: professor-neuromodulation
**Date**: 2026-05-09
**Anchors**: triage `docs/project/triage/20260509_1517_nmn_meta_continual_pivot.md` (Candidate D); paper-review §E (Doya 2002), §G (Lee 2024), §I (Rodriguez-Garcia 2026), §K (Tsuda 2021), §L (Wainstein 2025) in `docs/develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md`; `NMN_ARCHITECTURE_REVIEW.md` §2.2–§2.5, §3.1–§3.3; `project_plan.md` §3.2 (T/P split), §3.3 (opioid head).

**Verdict**: **run Candidate D on the current pre-Phase-0 architecture**, with one schedule modification (drop the third stage's hard return-to-stage-1 in favour of a **gradual→abrupt boundary contrast** plus a **double return**). Hand-off to `experiment-designer`.

**Biological-plausibility verdict**: **partial**. The current single-GRU modulator can in principle compute the stability-gap-attenuating gain operation (Rodriguez-Garcia 2026 §I.5) through `z_unimodal` and the `temperature` head, but it **cannot natively compute the Lee-2024 epistemic/aleatoric ratio** that drives autonomous boundary detection (§G.2). A null on Candidate D therefore has two distinct readings — a useful disambiguation that Phase 0 (T/P split, precision-head NLL) is built to address. Run Candidate D *first* as a floor: a positive result vindicates the architecture pre-Phase-0; a null sharpens what Phase 0 must fix.

---

## 1. Why a recurrent modulator with branched gain heads should attenuate the stability gap

Rodriguez-Garcia 2026 §I.5 (paper-review §I.5) shows that, under reparameterisation $\Phi(W) = g\,W$ with $g \geq 1$,

$$\nabla_W^2 \tilde{L}(W) = g^2 \, \nabla_{W_{\text{eff}}}^2 L(W_{\text{eff}}),
\qquad \lambda \to \lambda/g^2.$$

A *transient* gain spike at a task boundary flattens the loss-landscape curvature in the direction the boundary perturbs, so the SGD step at peak gain takes a smaller fraction of the local quadratic, and the post-switch survival dip shrinks. The mechanism is generic to any **multiplicative** scaling of effective weights; it is not specific to NA in particular.

The project's current architecture exposes this primitive at two of three injection sites (per `NMN_ARCHITECTURE_REVIEW.md` §2.2, §3.1, §3.3):

- **Injection A — `z_unimodal` (multiplicative mode).** $\gamma_A = \sigma(z_{\text{uni}}) \in (0,1)$ is the per-group multiplicative gate on the encoder's pre-fusion activations. Functionally, this gates the *effective input* into the task GRU, $x_{\text{proj}} \mapsto \sigma(z_{\text{uni}}) \odot x_{\text{proj}}$. By the same Hessian-scaling argument as §I.5, a transient excursion of $\sigma(z_{\text{uni}})$ at the boundary flattens curvature *along the encoder→task-GRU pathway*. Note: because $\sigma(\cdot) \in (0,1)$, the project's gain is **bounded above by 1**, not free to spike above the tonic baseline. The stability-gap effect therefore manifests as a *transient release* (gain rises toward 1 from a learned tonic offset that is typically below 1 because of the per-neuron `z_unimodal_baseline`), rather than as an above-baseline burst. This is biologically defensible — phasic NA in cortex is often modelled as transient *de-suppression* of cortical gain rather than super-baseline drive (Aston-Jones & Cohen 2005, paper-review §L.5 for the Wainstein formulation) — but it does mean the magnitude available to the project's modulator is roughly half what an unbounded $g \geq 1$ would have.
- **Injection C — `temperature` head.** $\tau_\pi = \mathrm{clip}(\mathrm{softplus}(z_{\text{act}}) + 0.5, 0.5, 3.0)$ scales policy-logit magnitude. In Doya's mapping (paper-review §E.1, §E.2), $\beta = 1/\tau_\pi$ is the inverse temperature — a noradrenergic exploration/exploitation knob. A boundary-triggered $\tau_\pi$ excursion (boundary → renewed exploration → wider action distribution) is exactly the Lee 2024 §G.3 prescription: at a context switch, push $\beta$ down so the agent re-explores rather than committing to the now-obsolete attractor. This is *not* a Hessian-flattening operation on the loss — it is a behavioural-exploration operation on the data distribution feeding the next gradient step. The two effects compound: the encoder gain shrinks the gradient norm at the boundary; the temperature spike enlarges the data-distribution support over which that gradient is averaged.
- **Injection B — memory `z_mem`.** This is *not* a gain-on-effective-weights operation; it is an additive bias on the task GRU's update gate (`NMN_ARCHITECTURE_REVIEW.md` §3.2, line 393). Its role for the stability-gap argument is different — by lowering the update gate at the boundary, the modulator *protects* the task GRU's hidden-state trajectory from the boundary perturbation, which is the Durstewitz 2025 §5 "fast / slow slot" decomposition in miniature.

**Net mechanistic claim**: the NMN should produce an attenuated stability-gap dip iff the modulator's hidden state $h_{\text{mod}}$ shifts at the boundary in a way that drives a *transient* movement of $\sigma(z_{\text{uni}})$ (toward 1), $\tau_\pi$ (away from 1), and $z_{\text{mem}}$ (toward update-gate suppression). The strong claim adds: the *speed* of this movement should be faster than the task GRU's own adaptation speed — otherwise the modulator is redundant with what the unmodulated baseline's recurrence already does (this is the addressee of the post-pivot architecture-question in `triage §6.4.1`).

---

## 2. Can the current single-GRU modulator compute the Lee-2024 boundary-detection signal?

Lee 2024 (paper-review §G.2–§G.3) drives the modulator from an **ensemble-derived epistemic/aleatoric uncertainty pair**:

$$\alpha = \frac{E}{E + A},
\qquad \beta \propto \frac{1}{\langle E \rangle},$$

with $E$ = variance across ensemble members of the mean Q-value, $A$ = mean across ensemble of within-member return variance. The boundary-detection mechanism in §G.5 is **distribution shift → epistemic-uncertainty spike → automatic $\alpha$ rise + $\beta$ drop**, with no explicit task-id signal. The agent re-explores because $E$ jumps, and forgets the obsolete value because $\alpha$ jumps.

The project's modulator is **not an ensemble**. It is a single GRU, and its inputs are `obs` (full vector, including `injury, satiation, nutrition`; `project_plan.md` §3.5). It therefore has no native handle on $E$ in the Lee form. Three weaker substitutes are available *given the present architecture*:

- **(i) Reward-prediction-error as a proxy for $E$.** A boundary-driven distribution shift will inflate the absolute TD error $|\delta_t|$ at the boundary. The modulator already conditions on the obs; if the value head's bootstrap target were also fed in, $|\delta|$ becomes computable inside the modulator. **It is not currently fed in.** A minimal change would expose the previous step's bootstrap residual to the modulator's GRU input. This is roughly the Doya-2002 §E.3 heuristic ("high $\delta$-variance → low $\beta$") instantiated as an architectural input rather than a hyperparameter rule.
- **(ii) Policy-entropy as a proxy for $E$ — Rodriguez-Garcia's choice.** Paper-review §I.3 uses $H(y_t) = -\sum_i \pi_i \log \pi_i$ as the gain-driver. This is *available* to a single-modulator architecture (no ensemble required) and is exactly the entropy that the project's actor head already computes for the PPO loss. **It is not currently fed back to the modulator.** Adding a one-step-lag $H_{t-1}$ as a modulator GRU input is a config-only change to the modulator's input vector.
- **(iii) Heteroscedastic precision-head NLL — the Phase 0 / `project_plan §3.4` direction.** The precision head trained on per-modality Kendall–Gal NLL gives the modulator a *teaching signal* for $\hat{\pi}_i$, and via Friston paper-review §F.3 the sustained reconstruction error $\overline{\varepsilon_i^2}$ is the natural analog of aleatoric uncertainty $A$ (paper-review §G.2: "expected uncertainty"). This is the load-bearing reason the project's Phase 0 includes a precision auxiliary; without it, the modulator has no biologically-coherent teaching signal for the ACh-like role at all (the `biological_plausibility_of_three_site_modulation.md` §1.2 verdict).

**Biological-plausibility critique**: route (iii) is the only one that produces a principled $E/A$ split. (i) and (ii) collapse onto a single "surprise"-like scalar that mixes epistemic and aleatoric uncertainty; the Lee-2024 separation is what makes the resulting $\alpha$ ratio bounded and stable. Without an aleatoric estimate, the "modulator increases learning rate at the boundary" mechanism in §G.5 has no normaliser, and the project's modulator has to learn that normalisation purely from the policy gradient — a much weaker signal. **Phase 0's T/P split + precision head is therefore the substrate-level minimum that recovers the Lee-2024 boundary-detection mechanism.** A null on Candidate D *under the current architecture* should not be read as "the architecture cannot do continual"; it should be read as "the architecture cannot do the *Lee*-flavoured continual mechanism, and is left to whatever proxy for surprise the modulator can extract from raw obs alone".

---

## 3. The falsifying measurement for catastrophic-forgetting resistance

Following the triage §4 DV (d) and Lee 2024 §G.5, the headline test is:

$$\Delta\text{forget} \;=\; \overline{\text{survival}}_{\mathcal{T}_1}\Big|_{\text{end stage 3}} \;-\; \overline{\text{survival}}_{\mathcal{T}_1}\Big|_{\text{end stage 1}},$$

with $\Delta\text{forget} \leq 0$ for any agent (post-stage-2 interference can only equal or hurt stage-1 retention). The *contrast* is

$$\Delta\Delta_{\text{forget}} \;=\; \Delta\text{forget}^{\text{mod}} \;-\; \Delta\text{forget}^{\text{unmod}},$$

with the prediction $\Delta\Delta_{\text{forget}} > 0$ (modulated forgets less; or equivalently, $|\Delta\text{forget}^{\text{mod}}| < |\Delta\text{forget}^{\text{unmod}}|$).

**Magnitude target.** Lee 2024 §G.5 (Fig. 3 in the paper; described in paper-review §G.5) shows the Doya-DaYu agent recovers post-switch performance roughly within 1–2 context durations ($M = 500$ steps), against ensemble-without-modulation baselines that take ≥ 5 context durations. Translated onto our continual schedule (1500-episode stages, ~80 steps/episode → ~120k steps/stage), a defensible threshold is:

- **Positive signature**: $|\Delta\text{forget}^{\text{mod}}| \leq \frac{1}{2} |\Delta\text{forget}^{\text{unmod}}|$, with $\geq 1\sigma$ across $n=3$ seeds. In survival-step units (Experiment 2's seed-noise floor $\sim 4$–$5$ steps), the unmodulated agent plausibly drops $\sim 10$–$20$ survival-steps after stage 2; the modulated agent should drop $\leq 5$–$10$.
- **Stability-gap (free-rider, DV (e)).** Define dip depth $d_k = \overline{\text{survival}}_{\text{stage }k\text{ asymptote}} - \min_{t \in [t_{\text{switch}}, t_{\text{switch}}+200\text{ ep}]} \overline{\text{survival}}_t$. Predict $d^{\text{mod}}_2 \leq \frac{1}{2} d^{\text{unmod}}_2$ (Rodriguez-Garcia 2026 §I.5 Fig. 4-style attenuation).
- **Mechanism check (free-rider).** At the boundary, the modulator's hidden state $h_{\text{mod}}$ should show a measurable transient — Mahalanobis distance $\| h_{\text{mod},t} - \bar{h}_{\text{mod, stage 1}} \|_{\Sigma^{-1}_{\text{mod, stage 1}}}$ should exceed 2 standard deviations within the first 50 episodes of stage 2, and $\sigma(z_{\text{uni}})$ should transiently rise toward 1, $\tau_\pi$ transiently away from 1. **Without this transient, any survival-level positive result is mechanistically unexplained.**

**Refutation**. If $\Delta\Delta_{\text{forget}} \leq 0$ *and* $h_{\text{mod}}$ shows no boundary-locked transient, the architecture is failing the floor test: even the cheapest stability-gap mechanism (the Hessian-flattening one, which only requires the gain heads to *move* at the boundary) is not engaged. That outcome — combined with a comparable null on the meta-context-conditioning probe (triage Candidate A) — is the project's signal that pre-Phase-0 is not a viable starting architecture for the headline pain-modulation work.

---

## 4. Schedule choice — why I depart slightly from Candidate D

The triage's three-stage active → passive → active schedule is *adequate* but not the cleanest test. Two recommendations:

**(a) Replace the third stage's hard return with a gradual–abrupt contrast.** Doya 2002 (paper-review §E.3) regulates $\beta$ from $Q$-value variance as a slow rule, and Sara 2009 (in BIB; cited by `project_plan.md` §3.2 as the rationale for $T \leftarrow P$ coupling) emphasises *phasic* NA bursts as the boundary detector. The cleanest contrast for adjudicating *which* timescale the modulator's gain is operating on is to compare an **abrupt** boundary (one episode boundary, full env swap) against a **gradual** boundary (linear interpolation of `predator.detection_range` across 200 episodes between stages) within the same experiment. Prediction: a fast-NA-burst modulator (entropy-driven, Rodriguez-Garcia §I.3) detects the abrupt boundary cleanly but produces *no* transient on the gradual boundary; a slow-tonic modulator (Doya §E.3 heuristic) produces a sustained drift on the gradual boundary. The current architecture's `mod_hidden_size = 16` GRU is timescale-ambiguous (no learned-$\tau$ filter yet — `project_plan.md` §3.1 says that's a Phase 0 add); this experiment *measures* which regime it is operating in. Useful evidence regardless of outcome.

**(b) Add a double return rather than a single return.** A schedule that visits stage 1 → stage 2 → stage 1 → stage 2 → stage 1 (five stages, each ~700 ep) gives **two** return episodes to stage 1, allowing measurement of whether forgetting is **monotone** (each return retains less than the previous — pure interference) or **non-monotone** (the second return *exceeds* the first — the modulator has built a re-usable context-specific subnetwork à la Tsuda 2021 hypertubes, paper-review §K.4). This is the cleanest possible architecture-vitality signal: monotone forgetting with shallow second-return is what an unmodulated GRU does; non-monotone with a *higher* second-return is the Tsuda-hypertube signature and is what the NMN literature claims is the architectural contribution. Zero engineering cost — it's still a stage list in the existing `--continual-schedule` YAML.

The combined schedule is: stage 1 (1500 ep, abrupt) → stage 2 (1500 ep) → stage 1 (700 ep, double-return probe) → stage 2 (700 ep, second exposure) → stage 1 (700 ep, second double-return) → finally a parallel cell with the same total budget but the stage 1↔2 boundary *gradual* over 200 ep. Approximately 1.5× the original Candidate D budget; still within the ≤ 10-cell ceiling (2 architectures × 3 seeds × 2 schedule variants = 12 cells; or trim to 1 schedule variant + 4 seeds = 8 cells if budget binds).

**Pre-condition check (configs only, no engineering).** The existing `train.py` continual mode (lines 135–195) accepts an arbitrary list of stage configs and episode boundaries, so the five-stage schedule is a YAML-only addition. Gradual interpolation between stages is *not* currently supported — it would need a small `developer` touch to interpolate per-step env params. **If the gradual-vs-abrupt arm requires engineering, drop it from the first round** and run only the five-stage abrupt-double-return schedule. The double-return is the stronger of the two adjudications.

---

## 5. Phase 0 architecture upgrades — what each adds, run-now vs. block

| Phase 0 upgrade | What it adds for Candidate D | Run-now-without-it? |
|---|---|---|
| **T/P split** (`project_plan §3.2`) | Gives separate fast/slow modulator timescales. The fast $h_P$ does the boundary-detection burst; the slow $h_T$ holds the persistent context. Without the split, a single GRU has to do both, and the *learned* timescale will sit at whichever dominates the gradient — typically the slow one, because the policy gradient backpropagated through long horizons washes out fast transients. **Strong expected improvement** on the stability-gap dip and the second-return retention. | Yes — but expect a muted result. The single GRU's most likely failure mode is that its `mod_h` time-constant is slow (~50 steps from `mod_hidden_size = 16` GRU dynamics under PPO horizons), so the boundary transient is smeared over ~50 episodes. |
| **Precision head (Kendall–Gal NLL)** (`project_plan §3.4`) | Gives $h_P$ a teaching signal for the aleatoric ($A$, paper-review §G.2) component of uncertainty. With this, the modulator can compute a Lee-2024-style $\alpha = E/(E+A)$ ratio (using sustained $\overline{\varepsilon^2}$ for $A$ and policy entropy for $E$). **Decisive for the Lee mechanism, weakly relevant for the Rodriguez-Garcia mechanism.** | Yes — Rodriguez-Garcia §I.5 alone (entropy-driven gain on the policy gradient) does not need the precision head. The lifelong-learning literature (Lee §G.5) does. Run Candidate D as a **Rodriguez-Garcia-flavoured** test under the current architecture; defer the Lee-flavoured test to Phase 0. |
| **Opioid head** (`project_plan §3.3`, on `o_noc`) | Targets nociception specifically, not generic context. **Not relevant to Candidate D.** The opioid head is for the chronic-pain absorbing-state test (H5), which is a Phase 4 question, not a continual-RL question. | Yes (irrelevant to this probe). |
| **Per-injection learnable-$\tau$ filter** (`project_plan §3.1`) | Lets the modulator output be slow at one site (e.g., B-memory) and fast at another (A-perception, C-temperature). Resolves the timescale ambiguity in (a) above without needing a full T/P split. **Moderate-to-strong improvement** for the gradual-vs-abrupt schedule. | Yes — the gradual-vs-abrupt schedule still measures whatever timescale the GRU happens to learn. Without the filter, all three sites share a single timescale; result interpretation is correspondingly coarser. |

**Recommendation: run on the current pre-Phase-0 architecture first.** Three reasons:

1. **A positive result on the current architecture is strong evidence the floor is intact** without the Phase 0 engineering bill (~2–3 weeks per `project_plan §3.2`). It shifts the probability mass of the v8 sensory-modulation null from "the architecture is broken" toward "the architecture works but the sensory-modulation question is a hard one".
2. **A null on the current architecture has two readable explanations** (single-GRU timescale collapse; missing aleatoric teaching signal), and both are concretely addressed by Phase 0 in known ways. The null *defines* what Phase 0 must repair, which is what the project needs from this probe more than a positive does.
3. **The probe is config-only** (modulo the optional gradual-stage developer touch), so the cost of running it pre-Phase-0 is small relative to the information it returns.

The alternative — block on Phase 0, then run — is defensible but costs 2–3 weeks before any data. The project's recent diagnostic series (v1–v8) has shown that running a probe and learning what it tells you is faster than designing a probe to be definitive in advance. Run now.

---

## 6. Hand-off

- **Memo location**: `docs/project/directions/nmn_continual_lifelong_probe.md` (this file).
- **Next agent**: **`experiment-designer`** to lock the YAML schedule (five-stage abrupt-double-return on the existing `--continual-schedule` machinery; optional gradual-arm contingent on a `developer` interpolator touch), the architecture cells (modulated vs. unmodulated, both pre-Phase-0), the seed budget ($n=3$ minimum, $n=4$ preferred per Experiment 2's noise floor), and the WandB tags. Bind to the per-stage segmented logging already in `train.py` line 1092.
- **Cross-references for the postdoc synthesis**:
  - `professor-rl-bayesian-dl` is owed the parallel meta-context memo (triage §6.4.1) for Candidate A. The two memos jointly answer "is the architecture broken or is the v8 question hard?" — this memo answers the continual half.
  - `professor-bayesian-brain` is the appropriate addressee if the postdoc decides to fold the Phase 0 precision-head argument (§2 route (iii) above) into a single combined Phase 0 case. The substrate-level claim is mine; the precision-as-gain mathematical justification is theirs.
- **Pain-modeling cross-link**: not load-bearing here (continual / lifelong is not a pain-construct question), but the opioid-head row in §5 above flags one cell of the §3 architecture that this probe does *not* test. That gap is `professor-pain-modeling`'s register, not mine.

## 7. Risk register

- **The single GRU's learned timescale is slow** → the boundary-locked transient in `h_mod` is smeared and the falsification check in §3 fails to register a transient even when the survival-level result is positive. Mitigation: report Mahalanobis distance over a *sliding window* of 50–200 episodes, not just a 50-episode window.
- **PPO truncated-BPTT washes out boundary transients** before they reach the gradient. The modulator's GRU sees boundaries through both its input (obs change) and the policy gradient (delayed). If truncation length is shorter than the boundary's settling time, the modulator can never *learn* to react to boundaries — it only sees within-segment dynamics. Verify truncation length against expected boundary-settling time before launch (likely a `developer` pre-flight check).
- **Survival-step seed noise** ($\sim 4$–$5$ steps from Experiment 2) may swallow the Lee-2024-magnitude effect (1–2 context durations of recovery). Mitigation: $n \geq 4$ seeds; report bootstrap CIs not point estimates.
- **The gradual-arm developer cost** is small but not zero. If the developer hand-off is non-trivial, drop the gradual arm and keep only the five-stage abrupt-double-return schedule for the first round. The double-return is the architecturally-decisive test; the gradual contrast is ancillary.
