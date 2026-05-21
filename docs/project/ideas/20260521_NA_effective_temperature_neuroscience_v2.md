---
title: "NA and the effective temperature — affective + computational neuroscience perspective (v2 stage)"
status: draft
author: top-level-claude (synthesis of user-Claude discussion, 2026-05-21)
audience: user + pi + research-postdoc
date: 2026-05-21
v_stage: v2
companions:
  - "docs/project/ideas/20260519_doya_modulation_extended_creative_ideas_v2.md"
  - "docs/project/ideas/20260521_doya_modulation_summary_v2.md"
  - "docs/project/ideas/20260521_5HT_effective_discount_neuroscience_v2.md"
  - "docs/project/ideas/20260521_ACh_effective_learning_rate_neuroscience_v2.md"
  - "docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md"
integration_status: "Standalone perspective memo, NOT yet integrated into the v2 extended ideas memo, the audit, or the lit-review folders. The user will integrate later. Parallel to the 5-HT and ACh effective-perspective memos."
one_line_summary: "Reframes NA's role from 'modulating the inverse-temperature parameter β' to 'gain modulation across cortical circuits with tonic-vs-phasic mode-switching' — better-grounded in affective and computational neuroscience and matching the FiLM-γ-at-logits architectural route."
---

## 1. What this doc is

A standalone perspective memo on the **NA → policy-temperature** knob of the project's neuromodulator-network framing, parallel in structure to the 5-HT effective-discount memo and the ACh effective-learning-rate memo (`20260521_5HT_effective_discount_neuroscience_v2.md`, `20260521_ACh_effective_learning_rate_neuroscience_v2.md`). It is **not** an integration into the v2 extended ideas memo, the audit, or the lit-review folders. It captures a single reframing: the *effective temperature* reading of NA is more biologically realistic than the *implementation temperature* reading the project has been using, and it strengthens the case for the FiLM-γ-at-logits architectural route.

A reader with basic RL only: in the textbook softmax policy $\pi(a|s) = \exp(\beta \cdot z_a(s)) / \sum_{a'} \exp(\beta \cdot z_{a'}(s))$, the inverse-temperature $\beta$ is a scalar that multiplies the action logits before the softmax. High $\beta$ → greedy / exploitative policy; low $\beta$ → stochastic / exploratory policy. Doya (2002) mapped this $\beta$ onto the noradrenergic (NA) system from the locus coeruleus. **This memo argues that the mapping is closer to the truth if we drop "scalar parameter on logits" and replace it with "gain modulation across cortical circuits with tonic-vs-phasic mode-switching."** The brain does not have a $\beta$; it has a locus coeruleus that releases NA broadly, modulating neuronal gain via α1, α2, and β-adrenergic receptors with different effects across regions and across firing modes. What looks like a "set the temperature" operation is an emergent consequence of how NA gain modulation reshapes the competition between candidate actions in motor and frontal circuits.

## 2. The shift in framing — implementation β versus effective temperature

The project's v2 extended memo handles NA via a rank-1 broadcast FiLM γ applied to the policy logits. This is algebraically identical to a context-conditioned softmax temperature, and the audit's NA verdict was that this is the cleanest case among the four Doya knobs. **This memo does not contradict that architectural claim; it strengthens it by showing that the FiLM-on-logits operation is structurally faithful to how NA actually works in the brain.**

The implementation-β reading says the modulator emits a scalar that gets multiplied into logits before the softmax. This is closer to the biology than the implementation-γ reading of 5-HT — the brain *does* have a gain-multiplying operation, and the locus coeruleus *does* broadcast a signal that effectively multiplies cortical responses. So the gap between implementation and biology is smaller here than for 5-HT or even for ACh.

But the gap is still meaningful in three places:

- The brain's gain modulation is not a single scalar across all units; it is receptor-specific and region-specific. α1, α2, and β-adrenergic receptors have different distributions and different effects, and the *net* gain modulation a given cortical area receives depends on which receptor subtype dominates locally. The "scalar β" reading collapses this heterogeneity.
- NA operates in **tonic and phasic modes** (Aston-Jones & Cohen 2005), and the behavioural temperature signature is qualitatively different between them. Tonic high-NA produces *exploration* (high effective temperature); phasic high-NA produces *exploitation* of the currently selected action (low effective temperature on that action specifically). The "single β" reading cannot capture this mode-switching.
- The temperature interpretation is downstream of *what gets selected*, not just *how confidently*. NA-induced gain on prefrontal representations changes which task-relevant feature dominates the action selection — modulating not just the entropy of the policy but its *direction*. This is a richer signature than scalar β captures.

**The effective-temperature reading keeps the FiLM-on-logits architecture but interprets it as gain modulation across the late motor/frontal stage of the action-selection circuit, with the temperature-like effect emergent from that gain pattern.** Mechanically the same forward-pass operation; conceptually a richer biological substrate.

## 3. Affective neuroscience grounding

Four lines of evidence ground the effective-temperature reading.

**Arousal and exploration in operant tasks.** Across many species and many tasks (Cohen, Aston-Jones, McGinty; older Yerkes-Dodson literature), arousal level — strongly correlated with tonic NA tone — modulates the exploration-exploitation balance. High arousal in novel or uncertain contexts produces increased behavioural variability; calm/focused arousal produces narrower, more decisive action selection. The relationship is the inverse-U "Yerkes-Dodson" curve: too low arousal → poor performance via under-engagement; too high arousal → poor performance via excessive distractibility. This is *exactly* what optimal-temperature analysis in RL predicts. The behavioural temperature is observed; the gain pattern in cortex is the underlying mechanism.

**Anxiety, stress, and policy stochasticity.** Acute stress and anxiety reliably increase behavioural variability and impair selective action — readings the temperature interpretation captures, the parameter interpretation does not. Clinical conditions characterised by NA dysregulation (PTSD, panic disorder, some anxiety subtypes) show the same pattern. Beta-blockers (β-adrenergic antagonists) used clinically for performance anxiety work by reducing peripheral NA effects; centrally-acting α2 agonists (clonidine, guanfacine) used for ADHD reduce tonic LC firing and improve focused exploitation. All of these only make sense under the effective-temperature reading.

**Damasio's somatic markers and exploration biases.** When interoceptive state signals high uncertainty about the environment (the somatic-marker signal is strong and contradicts current belief), the brain's downstream gain pattern reorganises toward exploration. This is structurally similar to the 5-HT effective-discount memo's somatic-marker argument, but for temperature rather than horizon. The interoceptive modulator c in the project's framing maps cleanly onto the somatic-marker signal that *induces* tonic-vs-phasic mode-switching in LC-NA, which *induces* the temperature-like behavioural signature.

**Aston-Jones & Cohen's adaptive gain theory.** The seminal computational-neuroscience interpretation of LC-NA function (Aston-Jones & Cohen 2005) is explicitly an *adaptive-gain* theory, not a temperature-setting theory. The LC-NA system has the *function* of regulating exploration-exploitation balance, but the *mechanism* by which it does so is gain modulation, which produces the temperature-like behavioural signature as a consequence. The temperature interpretation is downstream; gain modulation is upstream. This memo argues the project should adopt the upstream interpretation.

**Summary.** Across these four lines, behavioural and clinical neuroscience consistently treats effective temperature as a *measurable* consequence of arousal and gain modulation, not as a parameter the brain reads off and applies.

## 4. Computational neuroscience — three implementations the brain actually has

Three mechanisms by which the brain produces effective-temperature shifts without setting a scalar β.

**(1) Adaptive gain theory of LC-NA (Aston-Jones & Cohen 2005).** This is the canonical computational interpretation. The locus coeruleus operates in two modes: **tonic** (sustained moderate firing) and **phasic** (brief task-related bursts). Tonic firing produces broad cortical gain enhancement that increases distractibility and exploration; phasic firing produces a task-locked gain enhancement that focuses processing on the currently relevant feature. The mode-switching is itself regulated by reward expectations, conflict, and uncertainty signals from anterior cingulate and orbitofrontal cortex. The behavioural temperature signature emerges from which mode the LC is in and how strongly each is engaged.

This is the *most directly relevant* computational-neuroscience anchor for the project. The interoceptive modulator c is structurally analogous to the upstream signal (utility, conflict, uncertainty) that drives mode-switching in LC-NA; the FiLM γ output to policy logits is structurally analogous to the cortical gain pattern that the LC's mode produces. The mapping is no longer "we built the NA analogue;" it is "we instantiated the adaptive-gain mechanism Aston-Jones & Cohen identified."

**(2) Servan-Schreiber & Cohen gain-modulation model (1990).** An earlier and more mechanistic computational interpretation: NA modulates the *gain* of sigmoidal activation functions in cortical neurons, sharpening the input-output relationship. Higher gain → steeper sigmoid → more decisive responses → lower behavioural temperature. Lower gain → flatter sigmoid → more diffuse responses → higher behavioural temperature. This is precisely a FiLM-γ-style operation: a multiplicative modulation of the pre-activation that reshapes the response distribution.

For the project, this is the *most literally instantiated* mapping. A FiLM γ at the policy logits is mathematically a gain modulation of pre-activation logits. Servan-Schreiber & Cohen's model has been the canonical computational-cognitive interpretation of NA-gain modulation for three decades; the project's architecture is essentially the modern deep-RL instantiation of their idea.

**(3) Network reset theory (Sara 2009; Bouret & Sara 2005).** A third computational interpretation: phasic LC-NA bursts produce *network reset* — a brief destabilisation of current cortical activity patterns that allows reorganisation. The behavioural signature is exploration (the previous policy is reset, allowing alternative actions to compete for selection). This is conceptually different from gain modulation but produces a similar temperature-like effect. Under the network-reset reading, NA does not "raise the temperature smoothly" but "occasionally resets the policy distribution." 

For the project, this reading suggests the FiLM γ output should perhaps be more *transient* under high-uncertainty conditions than a steady scalar — the modulator could emit reset-like pulses rather than continuous gain modulation. This is a creative extension not currently in the v2 memo but worth flagging.

These three are complementary rather than competing; they describe different temporal scales and behavioural regimes of the same neuromodulator system. Across all three, the **scalar β-parameter** the implementation reading posits does not exist in any of the implementations.

## 5. The empirical anchor — Aston-Jones & Cohen 2005

The cleanest single computational-neuroscience anchor for the effective-temperature reading is **Aston-Jones & Cohen (2005), *An Integrative Theory of Locus Coeruleus-Norepinephrine Function: Adaptive Gain and Optimal Performance* (Annual Review of Neuroscience 28:403–450)**. They formalised the LC-NA system as implementing adaptive gain modulation that controls the exploration-exploitation trade-off via tonic-vs-phasic mode-switching. The paper:

- Synthesises decades of single-unit recordings in LC across primates and rodents.
- Connects LC firing patterns to cortical gain modulation via α1, α2, and β-adrenergic receptor distributions.
- Predicts the inverse-U relationship between LC tonic activity and task performance (Yerkes-Dodson).
- Identifies utility signals from anterior cingulate and orbitofrontal cortex as the upstream drivers of mode-switching.

The complementary behavioural anchor is **Yu & Dayan (2005), *Uncertainty, Neuromodulation, and Attention* (Neuron 46:681–692)** — the same paper anchored the ACh effective-learning-rate memo. Yu & Dayan formalised NA as the carrier of *unexpected uncertainty* (model-violating surprises), which gates exploration / network reset. The temperature-like behavioural effect emerges from this uncertainty signalling, not from a parameter setting.

The complementary classical-cognitive anchor is **Servan-Schreiber & Cohen (1990), *Dopamine, Frontal Cortex, and Schizophrenic Behavior: A Computational Model* (Annual Review of Neuroscience)** — earlier work that established the gain-modulation interpretation, then later extended to NA in the cortico-frontal circuits.

Taken together: the LC-NA system implements adaptive gain via mode-switching driven by uncertainty signals; the FiLM-γ-at-logits architecture is the deep-RL instantiation of the same computation.

## 6. Implications for the project's framing

Four implications.

**(a) The FiLM-on-logits architecture is biologically faithful, not merely convenient.** The audit's NA Candidate 1 (rank-1 broadcast FiLM γ at policy logits) is currently presented as the simplest case among the four Doya knobs — algebraically clean, architecturally trivial. Under the effective-temperature reading, **it is also the biologically correct instantiation** of the Servan-Schreiber & Cohen gain-modulation mechanism that has been the canonical NA-cognition interpretation since 1990. The "simplest" claim is also the "most faithful" claim.

**(b) Mode-switching suggests a temporal extension to FiLM γ.** The Aston-Jones & Cohen tonic-vs-phasic distinction implies that the modulator should not just emit a steady FiLM γ; it should also produce transient, task-locked pulses when uncertainty signals demand them. The v2 memo's NA Candidate 1 does not currently encode this. A creative extension worth integrating: the modulator's Head 1 output could have a *phasic* component (event-locked pulses on detected uncertainty) layered on top of its *tonic* component (the steady FiLM γ). This is a small architectural addition with strong neuroscience backing.

**(c) The CAT-SAC rejection pattern is partially addressed by this framing.** CAT-SAC (ICLR 2021 reject) was rejected partly for "weak theoretical foundation" — the per-state entropy modulation was justified by intuition rather than a derivation. Under the effective-temperature reading, the FiLM-γ-at-logits operation *is* derivable from the Servan-Schreiber & Cohen gain-modulation framework, which is itself derivable from biophysical models of cortical neurons under adrenergic modulation. The derivation chain runs:
> Biophysical → Servan-Schreiber gain → Aston-Jones adaptive gain → behavioural temperature

This is not a parameter-modulation derivation but a *gain-modulation* derivation that *induces* temperature as a behavioural signature. A reviewer asking "why per-state temperature?" gets a substantive answer.

**(d) Construct validity is strongest of the four knobs.** Among NA, ACh, DA, and 5-HT, the effective-perspective memos collectively suggest that **NA has the strongest case for biological faithfulness** — the FiLM-γ-at-logits architecture is the direct deep-RL instantiation of a 35-year-old canonical cognitive-neuroscience model (Servan-Schreiber & Cohen 1990). The construct-validity argument here is essentially mature; the project does not need to *make a case*, only *cite the case that exists*.

## 7. Pointers and integration status

This is a **standalone perspective memo**. It does not modify the v2 extended ideas memo, the audit, the audit's story doc, the v1 compact memo, the v2 summary, or the lit-review folders.

- **V2 extended creative ideas memo (the doc this reframing is most directly relevant to)**: [`docs/project/ideas/20260519_doya_modulation_extended_creative_ideas_v2.md`](20260519_doya_modulation_extended_creative_ideas_v2.md) — Section 3 (NA) is the section that would gain a neuroscience-anchor preface under integration, plus the phasic-mode extension as a fourth candidate.
- **V2 summary memo**: [`docs/project/ideas/20260521_doya_modulation_summary_v2.md`](20260521_doya_modulation_summary_v2.md) — the NA row of the per-knob candidate table is the part that would shift.
- **5-HT effective-discount perspective (sibling memo)**: [`docs/project/ideas/20260521_5HT_effective_discount_neuroscience_v2.md`](20260521_5HT_effective_discount_neuroscience_v2.md).
- **ACh effective-learning-rate perspective (sibling memo)**: [`docs/project/ideas/20260521_ACh_effective_learning_rate_neuroscience_v2.md`](20260521_ACh_effective_learning_rate_neuroscience_v2.md) — the three sibling memos collectively reinterpret the NA / ACh / 5-HT knobs through the effective-property lens.
- **Theoretical audit (NA verdict)**: [`docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md`](../critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md) — NA was already judged the cleanest case; this memo strengthens that judgment by adding the biological-faithfulness argument.
- **Temperature lit-review folder**: `docs/project/references/Temperature/` — the four lit-review files there (SAC foundations, SAC variants, softmax operators, KL/Munchausen) ground the architectural side. The neuroscience anchors here (Aston-Jones & Cohen 2005, Yu & Dayan 2005, Servan-Schreiber & Cohen 1990, Sara 2009, Bouret & Sara 2005) are not in those reviews and would need to be added under integration.

**For the integration the user will do later:** the natural update is to add a one-paragraph neuroscience-anchor preface to the v2 memo's Section 3 (NA) noting that the FiLM-γ-at-logits Candidate 1 *is* the modern deep-RL instantiation of Servan-Schreiber & Cohen's 1990 gain-modulation model, and to add a fourth candidate (Phasic-mode extension) capturing the Aston-Jones & Cohen tonic-vs-phasic distinction. The neuroscience-anchor references belong in the Temperature lit-review folder as a fifth review file or in a sibling neuroscience-anchors folder.
