---
title: "Doya modulation via FiLM/hypernet — integrated v3 (concise)"
status: draft (v3 concise; feedback round before v3 extensive)
author: top-level-claude (integration synthesis)
audience: user + pi + research-postdoc
date: 2026-05-21
v_stage: v3 (concise)
supersedes_at_concept_level:
  - "v1 compact ideas memo (20260518_doya_modulation_via_film_hypernet_implementation_ideas.md)"
  - "v2 extended creative ideas memo (20260519_doya_modulation_extended_creative_ideas_v2.md)"
  - "v2 summary memo (20260521_doya_modulation_summary_v2.md)"
note_on_supersession: "v2 docs are not archived. They remain canonical for their respective scopes (compact / extended / summary). v3 integrates the four sibling effective-perspective memos and the three-professor AI-implementation opinions on top of v2's architectural skeleton, AND commits to the pain grid world as the experimental environment."
companions:
  - "docs/project/ideas/20260521_NA_effective_temperature_neuroscience_v2.md"
  - "docs/project/ideas/20260521_ACh_effective_learning_rate_neuroscience_v2.md"
  - "docs/project/ideas/20260521_DA_effective_RPE_gain_neuroscience_v2.md"
  - "docs/project/ideas/20260521_5HT_effective_discount_neuroscience_v2.md"
  - "docs/project/ideas/20260521_AI_implementation_opinions_v2.md"
  - "docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md"
one_line_summary: "Integrated v3 (concise) of the NMN-FiLM-hypernet idea — the project's claim under the effective-perspective reframing, the pain grid world as the experimental environment, three-discipline AI feasibility, and the canonical four-knob multi-signature foraging + threat-avoidance experiment. Concise; feedback round precedes v3 extensive."
---

## 1. Plain-language entry point

This is the **integrated v3** of the project's NMN-FiLM-hypernet idea. It supersedes the v2 trio (extended ideas memo, summary memo, audit) at the *conceptual* level by folding in the four sibling effective-perspective memos and the three-professor AI-implementation opinions. **The v2 docs remain canonical for their compact / extended / summary scopes; v3 is the integrated story.**

What changed:

- **The project's claim is now effective-perspective.** The modulator does *not* set Doya 2002's hyperparameters directly. It induces FiLM/hypernet-style forward-pass conditioning that produces *behavioural signatures* matching what experiments measure as temperature, learning rate, RPE gain, and discount.
- **The environment is the pain grid world.** Foraging + threat avoidance is the ecologically valid paradigm for neuromodulator research. MiniGrid was the textbook isolation environment three professors recommended; the pain grid world is the project's home environment and is strictly better for cross-knob multi-signature observation.
- **AI feasibility has been audited from three disciplines.** DL-theory, RL, and Bayesian-NN professors converged on which knobs are implementable today and which require novel architectural assembly.

The paper this v3 sketches is **exploratory**: observe hyperparameter-modulation-like phenomena emerge from FiLM/hypernetwork architectures in the pain grid world, with behavioural signatures that bridge to top-venue neuroscience anchors per knob. Not a strict replication of Doya 2002; a generative observation that current AI architectures can produce phenomena matching what affective and computational neuroscience study in animals.

## 2. Why v3 supersedes v2 (the headline shift)

| Aspect | v2 (extended + summary) | v3 (integrated) |
|---|---|---|
| Framing of γ, α, β, κ | Implementation parameters | **Effective measurable properties** |
| 5-HT route | γ-conditional UVFA with Head 2 emitting γ_B(c) | **FiLM-only effective horizon** — Head 2 optional |
| ACh route | Chain-rule channel as clever architectural identity | **Chain-rule channel = biologically correct mechanism** |
| NA route | FiLM-γ at logits as simplest case | **FiLM-γ at logits = direct deep-RL instantiation of Servan-Schreiber & Cohen 1990** |
| DA route | Scalar κ baseline + richer candidates | **Distributional DA (Dabney 2020 *Nature*) = baseline; scalar κ = degenerate special case** |
| Audit gauge concern | Scalar Head 2 for DA gauge-equivalent to ACh-α | **Gauge dissolves under distributional DA reading** |
| Environment | Unspecified | **Pain grid world (foraging + threat avoidance)** |
| Construct-validity story | Doya analogy | **Biological-faithfulness via top-venue neuroscience anchors** |

## 3. The two-headed modulator (brief recap)

```
                 context c (interoceptive state)
                            |
                     [ modulator m_φ ]
                       /          \
                Head 1              Head 2
        (FiLM γ, β) → activations  (vectors / functions) → loss / return
                |                       |
            forward pass           backward / return-estimator
```

Head 1 modifies hidden activations via FiLM (multiplicative scale + additive shift). By chain rule, it also gates the backward pass. Head 2 emits structured outputs (vectors, distortion preferences, optionally scalars) into the loss and return estimator. The v3 architectural conviction: **vector / function outputs from Head 2 are the biological baseline, not richer alternatives.** Scalar outputs are degenerate special cases.

## 4. The four knobs under the effective reading (concise)

**NA — effective temperature.** The brain implements adaptive gain modulation across cortical circuits with tonic-vs-phasic mode-switching (Aston-Jones & Cohen 2005 *Annu. Rev. Neurosci.*). FiLM-γ-at-policy-logits *is* the modern deep-RL instantiation of Servan-Schreiber & Cohen's 1990 gain-modulation model. **Observable in pain grid world:** Yerkes-Dodson inverse-U in task performance vs. learned γ_NA(c) under threat-proximity conditions; policy entropy decreases monotonically in γ_NA(c). **Feasibility:** feasible today (all three disciplines).

**ACh — effective learning rate.** The brain implements expected-uncertainty precision-gating on bottom-up signals (Yu & Dayan 2005 *Neuron*). The FiLM chain-rule channel from the audit's 2026-05-18 addendum is the biologically correct mechanism — the coupling between feature-gating and gradient-gating is *the right biology*, not a caveat. **Observable in pain grid world:** stratified-direction gradient probe across upstream parameter strata — preserved within FiLM-channel strata, rotated across strata; behavioural learning-rate shifts under novel threat conditions. **Feasibility:** feasible today (DL-theory, Bayesian-NN); feasible-but-novel under standard RL identifiability tests.

**DA — effective TD-error gain.** The brain implements a distributional population code over RPE quantiles (Dabney *et al.* 2020 *Nature* — different DA neurons code different quantiles of the return distribution) plus opponent two-channel routing (Collins & Frank OpAL D1/D2) plus tonic-vs-phasic vigor (Niv *et al.* 2007). **The Dabney 2018 ICML (IQN algorithm) and Dabney 2020 *Nature* (brain mechanism) unification is the single strongest algorithm-to-biology bridge in the project's framing.** Scalar κ from Head 2 is the *outdated* reading; distributional output via IQN-distortion β(c) is the biological baseline. The audit's DA/ACh gauge concern *dissolves* because biologically faithful DA Head 2 was never scalar. **Observable in pain grid world:** per-context return-distribution shape; choice probability between low-variance and high-variance reward paths as a function of interoceptive context (replicates Dabney 2020 mouse paradigm). **Feasibility:** feasible today / feasible-but-novel — strongest cross-disciplinary case.

**5-HT — effective discount.** The brain implements timescale-mixture gating across the cortical hierarchy (Hasson, Murray, Chaudhuri timescale hierarchy; Tanaka 2007 *Nature Neuroscience* 5-HT depletion shifts activity between short- and long-timescale regions). **There is no neural γ-multiplication step.** FiLM-only suffices — Head 2 emitting γ_B(c) is the *algorithmic-control variant*, not the canonical implementation. **Observable in pain grid world:** calibrated effective-horizon shift via reward-delay perturbation across interoceptive contexts; effective discount inferred from log|ΔV| slope vs. delay k. **Feasibility:** feasible today under FiLM-only reading.

## 5. The canonical v3 experiment — multi-knob foraging + threat-avoidance in the pain grid world

**Setup.** Pain grid world configured with two food-acquisition paths: a low-variance steady-reward path and a high-variance large-reward path with occasional pain. Interoceptive state c = (satiation level, recent pain exposure, optionally fatigue). Modulator m_φ ingests c, emits Head 1 (FiLM γ, β on critic + policy hidden activations) and Head 2 (context-conditioned IQN distortion β(c) on the return-distribution head; optional γ_B(c) for 5-HT, scalar w_r(c) for DA-vector route).

**Observable signatures, one per knob, in the same training run:**

| Knob | Signature | Neuroscience anchor |
|---|---|---|
| **NA** | Inverse-U Yerkes-Dodson curve in task return vs. inferred γ_NA(c) under threat proximity | Aston-Jones & Cohen 2005 |
| **ACh** | Stratified-direction gradient probe shows within-stratum preservation and across-stratum rotation under novel-threat exposure | Yu & Dayan 2005 |
| **DA** | Per-context return-distribution shape (quantile emission curves) shifts across satiation × threat states | Dabney 2020 *Nature* |
| **5-HT** | Calibrated effective horizon shifts with recent pain exposure (log\|ΔV\|-slope test) | Tanaka 2007 |

**Baselines.** (1) Vanilla agent (no modulator). (2) Agent with interoceptive state as observation only (no modulator head). (3) Full FiLM-modulator agent (Head 1 only). (4) Full two-headed modulator agent (Head 1 + Head 2). Comparison addresses the CAT-SAC ICLR 2021 reviewer pattern — *"why not just add c to observations?"* — directly.

**Statistical rigour.** ≥10 seeds. Effect-size + significance tests, not just curves. Long enough training to reach behavioural plateau before measuring signatures.

**What this experiment is the bridge to.** Berridge wanting/liking under different motivational states, Aston-Jones LC firing under threat, Dabney 2020 mouse reward-prediction, Tanaka 2007 fMRI under 5-HT depletion — all of these were studied in foraging + threat-avoidance paradigms. The pain grid world IS that paradigm in deep-RL form.

## 6. AI-feasibility convergence (three disciplines)

| Knob | DL-theory | RL | Bayesian-NN |
|---|---|---|---|
| NA | feasible today | feasible today | feasible today |
| ACh | feasible today | feasible-but-novel | feasible today (cleanest story) |
| DA | feasible-but-novel | feasible today, strongest precedent | feasible today (richest mid-novelty) |
| 5-HT | feasible today | feasible-but-novel | feasible-but-novel |

The three-professor convergence: the canonical experiment can be implemented today on the pain grid world with PPO or SAC backbone, IQN return head, FiLM modulator on critic and policy hidden states, plus a Bayesian audit (last-layer Laplace on the critic) for the DA distributional read-out. No exotic primitives.

## 7. What v3 claims (the publishable headline)

> **FiLM/hypernetwork modulators conditioned on interoceptive state produce hyperparameter-modulation-like behavioural signatures across all four Doya neuromodulator-to-RL-hyperparameter mappings, in a foraging + threat-avoidance environment, with each signature mapping onto a top-venue neuroscience anchor.**

What this claims:
- Existing AI architectures (FiLM + hypernet + standard RL) can produce the *phenomena* the neuromodulator literature measures.
- The bridge from algorithm to biology is via *observed behavioural signatures*, not via parameter replication.
- The pain grid world is sufficient to display all four knobs simultaneously.

What this does NOT claim:
- That the FiLM modulator *is* the biological neuromodulator system.
- That the project has derived (from first principles) why interoceptive state c modulates each knob.
- That the construct-validity story is closed.

## 8. What v3 still owes — open questions for v3 extensive

The honest gaps the v3 concise leaves for the extensive version to address:

1. **Construct-validity derivation.** Why *should* interoceptive state c modulate these knobs in these ways? Right now the answer is "because the neuroscience literature observes it." A derived answer (from Bayesian / predictive-coding / active-inference principles) would be stronger.
2. **The CAT-SAC failure mode.** ICLR 2021 rejected CAT-SAC for weak theoretical foundation and unconvincing empirical results on hard exploration. v3 inherits the empirical-rigour requirement; the four-baseline design plus ≥10 seeds is the project's defence.
3. **The two-headed-vector-or-function-output finding.** The audit + the four sibling memos collectively recommend that Head 2 outputs be vector or function-valued rather than scalar. v3 architectural design needs to commit to specific output dimensionalities per knob.
4. **The pain-grid-world parameterisation for clean isolation experiments.** Each knob needs an isolation configuration (one knob active, others held constant) in addition to the full multi-knob setting.
5. **Reproducibility plan.** WandB-archived runs + supplementary environment code + statistical rigour beyond CAT-SAC's failure thresholds.
6. **Integration of the four lit-review folders into the references section.** The four sibling memos cite their respective lit-review folders' anchors; v3 needs to surface the cross-knob citations cleanly.

## 9. Pointers (cross-links to all integrated sources)

- **Effective-perspective sibling memos (the neuroscience grounding):**
  - [NA](20260521_NA_effective_temperature_neuroscience_v2.md)
  - [ACh](20260521_ACh_effective_learning_rate_neuroscience_v2.md)
  - [DA](20260521_DA_effective_RPE_gain_neuroscience_v2.md)
  - [5-HT](20260521_5HT_effective_discount_neuroscience_v2.md)
- **AI-implementation opinions (three disciplines):** [`docs/project/ideas/20260521_AI_implementation_opinions_v2.md`](20260521_AI_implementation_opinions_v2.md)
- **V2 extended creative ideas memo (the architectural source):** [`docs/project/ideas/20260519_doya_modulation_extended_creative_ideas_v2.md`](20260519_doya_modulation_extended_creative_ideas_v2.md)
- **V2 summary memo:** [`docs/project/ideas/20260521_doya_modulation_summary_v2.md`](20260521_doya_modulation_summary_v2.md)
- **Theoretical audit and its addenda:** [`docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md`](../critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md)
- **Audit story doc (reader-facing audit arc):** [`docs/project/critiques/20260518_film_as_hyperparameter_modulator_audit_story.md`](../critiques/20260518_film_as_hyperparameter_modulator_audit_story.md)
- **Four lit-review folders:** `docs/project/references/{Temperature, Learning_Rate, TD, Gamma}/`

---

**Feedback round.** This is the v3 concise. The user has explicitly indicated they will provide feedback before the v3 extensive is drafted. Suggested feedback prompts to consider:

- Is the effective-perspective claim correctly summarised?
- Is the pain grid world commitment positioned correctly relative to the professors' MiniGrid recommendation?
- Are the four observable signatures the right ones to commit to? Any to drop or add?
- Is the "what v3 claims / does not claim" honesty split right?
- Which of the six open questions should v3 extensive prioritise resolving?
- What's missing that should be in the v3 extensive but is absent from this concise?
